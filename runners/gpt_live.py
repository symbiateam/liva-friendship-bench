from glob import glob
import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

try:
    import websockets
except ImportError:
    sys.exit("Missing dependency: pip install websockets")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_context import get_system_prompt


WS_URL = "wss://api.openai.com/v1/live/sessions"
MODEL = "gpt-live-1"
_DEFAULT_BACKEND_MODEL = "gpt-5.6-terra"
_DEFAULT_REASONING = "low"
_DEFAULT_VOICE = "marin"
_DEFAULT_TASKS = ["emotion_mirroring"]

SAMPLE_RATE = 24_000
CHUNK_SIZE = 1024
FRAME_DUR = CHUNK_SIZE / SAMPLE_RATE
GRACE_AFTER_INPUT = 30.0
START_TIMEOUT = 20.0
CLOSE_TIMEOUT = 15.0
SPEECH_RMS = 0.01

DEBUG = False
_EOS = object()


def dprint(msg):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def _load_dotenv():
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _parse_args():
    parser = argparse.ArgumentParser(description="GPT-Live-1 inference")
    parser.add_argument("--root-dir", required=True, help="Root directory containing task subdirectories")
    parser.add_argument("--task", default=None, nargs="*", help="Task name(s). Default: emotion_mirroring")
    parser.add_argument("--prefix", default="", help="Prefix for input wav files")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent Live sessions (tier 1 allows 25)")
    parser.add_argument("--voice", default=_DEFAULT_VOICE)
    parser.add_argument(
        "--delegation",
        choices=["responses", "none"],
        default="responses",
        help="responses = Terra backend (OpenAI's FDB setup); none = bare voice model",
    )
    parser.add_argument("--backend-model", default=_DEFAULT_BACKEND_MODEL)
    parser.add_argument(
        "--reasoning",
        default=_DEFAULT_REASONING,
        help="Backend reasoning effort (minimal|low|medium|high). '' to omit.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        type=str,
        help="Override conversation instructions. '' for no instructions field.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _mono(sig: np.ndarray) -> np.ndarray:
    return sig if sig.ndim == 1 else sig.mean(axis=1)


def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return sig
    wav_f32 = torch.from_numpy(sig.astype(np.float32) / 32768.0).unsqueeze(0)
    wav_rs = AF.resample(wav_f32, orig_sr, target_sr)
    return (wav_rs.squeeze().numpy() * 32768).astype(np.int16)


def _chunk(sig: np.ndarray, frame_len: int) -> List[np.ndarray]:
    pad = (-len(sig)) % frame_len
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, dtype=sig.dtype)])
    return [sig[i : i + frame_len] for i in range(0, len(sig), frame_len)]


def _b64(pcm: np.ndarray) -> str:
    return base64.b64encode(pcm.tobytes()).decode("utf-8")


def _speech_stats(audio: np.ndarray, sr: int):
    frame = int(sr * 0.02)
    x = audio.astype(np.float32) / 32768.0
    onset, voiced = None, 0
    for i in range(0, len(x) - frame + 1, frame):
        if np.sqrt(np.mean(x[i : i + frame] ** 2)) > SPEECH_RMS:
            voiced += 1
            if onset is None:
                onset = round(i / sr, 3)
    return onset, round(voiced * 0.02, 3)


def _session_config(instructions, args):
    session = {
        "model": MODEL,
        "audio": {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "output": {"voice": args.voice},
        },
    }
    if instructions:
        session["instructions"] = instructions
    if args.delegation == "none":
        session["delegation"] = {"type": "client"}
    else:
        responses = {"model": args.backend_model}
        if args.reasoning:
            responses["reasoning"] = {"effort": args.reasoning}
        session["delegation"] = {"type": "responses", "responses": responses}
    return session


async def stream_wav(input_path: Path, output_path: Path, instructions: str, args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    wav, sr = sf.read(str(input_path), dtype="int16")
    wav = _resample(_mono(wav), sr, SAMPLE_RATE)
    frames = _chunk(wav, CHUNK_SIZE)
    expected = len(wav)
    input_dur = expected / SAMPLE_RATE
    silence_b64 = _b64(np.zeros(CHUNK_SIZE, np.int16))

    audio_out_q: asyncio.Queue = asyncio.Queue()
    out_chunks: List[np.ndarray] = []
    transcript: List[str] = []
    metrics = {
        "model": MODEL,
        "delegation": args.delegation,
        "backend_model": args.backend_model if args.delegation != "none" else None,
        "reasoning_effort": (args.reasoning or None) if args.delegation != "none" else None,
        "voice": args.voice,
        "input_sec": round(input_dur, 3),
        "stream_start_sec": None,
        "speech_onset_sec": None,
        "voiced_sec": 0.0,
        "timed_out": False,
        "delegations": 0,
        "usage_seconds": None,
        "session_id": None,
        "error": None,
    }

    started = asyncio.Event()
    done = asyncio.Event()
    closed = asyncio.Event()
    st = {"input_end": None}

    async with websockets.connect(
        WS_URL,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        max_size=None,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.start",
                    "event_id": "event_start",
                    "session": _session_config(instructions, args),
                }
            )
        )

        async def listener():
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    t = msg.get("type", "")
                    dprint(f"recv: {t}")
                    if t == "session.started":
                        metrics["session_id"] = (msg.get("session") or {}).get("id")
                        started.set()
                    elif t == "session.output_audio.delta":
                        await audio_out_q.put(base64.b64decode(msg["delta"]))
                    elif t == "session.output_transcript.delta":
                        transcript.append(msg.get("delta", ""))
                    elif t == "session.delegation.created":
                        metrics["delegations"] += 1
                    elif t == "session.usage.updated":
                        metrics["usage_seconds"] = (msg.get("usage") or {}).get("seconds")
                    elif t == "session.closed":
                        metrics["usage_seconds"] = (msg.get("usage") or {}).get("seconds", metrics["usage_seconds"])
                        closed.set()
                        break
                    elif t == "error":
                        metrics["error"] = msg.get("error", msg)
                        print(f"[ERROR] Live API error: {metrics['error']}", file=sys.stderr)
            except websockets.exceptions.ConnectionClosed:
                dprint("WebSocket closed")
            finally:
                started.set()
                done.set()
                closed.set()
                await audio_out_q.put(_EOS)

        async def sender():
            try:
                for f in frames:
                    await ws.send(json.dumps({"type": "session.input_audio.append", "audio": _b64(f)}))
                    await asyncio.sleep(FRAME_DUR)
                st["input_end"] = time.time()
                while not done.is_set():
                    await ws.send(json.dumps({"type": "session.input_audio.append", "audio": silence_b64}))
                    await asyncio.sleep(FRAME_DUR)
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                pass

        async def writer(t0: float):
            total = 0
            got_any = False
            last_data = t0
            while total < expected:
                try:
                    data = await asyncio.wait_for(audio_out_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    data = None
                now = time.time()
                if data is _EOS:
                    break
                if data is None:
                    if not got_any and st["input_end"] is not None and now - st["input_end"] >= GRACE_AFTER_INPUT:
                        metrics["timed_out"] = True
                        break
                    if got_any and now - last_data >= GRACE_AFTER_INPUT:
                        metrics["error"] = metrics["error"] or "output stream stalled"
                        break
                    continue
                pcm = np.frombuffer(data, dtype=np.int16)
                if not pcm.size:
                    continue
                last_data = now
                if not got_any:
                    got_any = True
                    metrics["stream_start_sec"] = round(now - t0, 3)
                remain = expected - total
                pcm = pcm[:remain]
                out_chunks.append(pcm)
                total += pcm.size

        listener_t = asyncio.create_task(listener())
        try:
            await asyncio.wait_for(started.wait(), START_TIMEOUT)
        except asyncio.TimeoutError:
            listener_t.cancel()
            raise RuntimeError("session.started not received")
        if metrics["session_id"] is None:
            listener_t.cancel()
            raise RuntimeError(f"session.start rejected: {metrics['error']}")

        t0 = time.time()
        sender_t = asyncio.create_task(sender())
        await writer(t0)
        done.set()
        sender_t.cancel()
        try:
            await sender_t
        except asyncio.CancelledError:
            pass

        try:
            await ws.send(json.dumps({"type": "session.close"}))
            await asyncio.wait_for(closed.wait(), CLOSE_TIMEOUT)
        except Exception as e:
            dprint(f"close: {e}")
        listener_t.cancel()
        try:
            await listener_t
        except asyncio.CancelledError:
            pass

    audio = np.concatenate(out_chunks) if out_chunks else np.zeros(0, np.int16)
    if audio.size < expected:
        audio = np.concatenate([audio, np.zeros(expected - audio.size, np.int16)])
    sf.write(str(output_path), audio, SAMPLE_RATE, subtype="PCM_16")

    onset, voiced = _speech_stats(audio, SAMPLE_RATE)
    metrics["speech_onset_sec"] = onset
    metrics["voiced_sec"] = voiced
    metrics["transcript"] = "".join(transcript).strip()
    with open(output_path.parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


async def run_all(jobs, args):
    sem = asyncio.Semaphore(args.concurrency)

    async def one(input_p: Path, output_p: Path, instructions: str):
        async with sem:
            print(f"[RUN] {input_p} -> {output_p}", flush=True)
            try:
                await stream_wav(input_p, output_p, instructions, args)
                m = json.load(open(output_p.parent / "metrics.json"))
                print(
                    f"[DONE] {output_p}  onset={m['speech_onset_sec']}s voiced={m['voiced_sec']}s "
                    f"timed_out={m['timed_out']} delegations={m['delegations']} "
                    f"usage={m['usage_seconds']}s",
                    flush=True,
                )
            except Exception as e:
                print(f"[ERROR] {input_p}: {e}", flush=True)
                if output_p.exists():
                    output_p.unlink()

    await asyncio.gather(*(one(*j) for j in jobs))


def main():
    global DEBUG
    args = _parse_args()
    DEBUG = args.debug
    _load_dotenv()
    tasks = args.task if args.task else _DEFAULT_TASKS

    all_wav_files = []
    for task in tasks:
        all_wav_files.extend(sorted(glob(f"{args.root_dir}/{task}/*/{args.prefix}input.wav")))
    if not all_wav_files:
        all_wav_files = sorted(glob(f"{args.root_dir}/*/{args.prefix}input.wav"))
    all_wav_files.sort(
        key=lambda p: (str(Path(p).parent.parent), int(Path(p).parent.name) if Path(p).parent.name.isdigit() else 0)
    )
    if args.limit:
        all_wav_files = all_wav_files[: args.limit]

    current_task = tasks[0] if len(tasks) == 1 else None
    jobs = []
    for inp in all_wav_files:
        input_p = Path(inp)
        output_p = Path(inp.replace("input.wav", "output.wav"))
        if not args.overwrite and output_p.exists():
            print(f"[SKIP] {output_p} already exists")
            continue
        if args.system_prompt is not None:
            instructions = args.system_prompt
        else:
            task = current_task or input_p.parent.parent.name
            instructions = get_system_prompt(str(input_p.parent), task)
        jobs.append((input_p, output_p, instructions))

    print(
        f"[INFO] {len(jobs)} samples, model={MODEL}, delegation={args.delegation}, "
        f"backend={args.backend_model if args.delegation != 'none' else None}, "
        f"reasoning={args.reasoning or None}, concurrency={args.concurrency}"
    )
    asyncio.run(run_all(jobs, args))


if __name__ == "__main__":
    main()
