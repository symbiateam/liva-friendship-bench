import concurrent.futures as cf, glob, json, os, sys, time, requests

KEY = os.environ.get("ELEVENLABS_API_KEY") or sys.exit("Set ELEVENLABS_API_KEY")


def run(p):
    o = p.replace("output.wav", "output_scribe.json")
    if os.path.exists(o):
        return
    for a in range(4):
        try:
            r = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": KEY},
                data={
                    "model_id": "scribe_v2",
                    "timestamps_granularity": "word",
                    "diarize": "false",
                    "tag_audio_events": "true",
                    "language_code": "eng",
                },
                files={"file": open(p, "rb")},
                timeout=300,
            )
            if r.ok:
                json.dump(
                    [
                        dict(w=w.get("text"), s=w.get("start"), e=w.get("end"), type=w.get("type"))
                        for w in r.json().get("words", [])
                    ],
                    open(o, "w"),
                )
                return
        except Exception:
            pass
        time.sleep(3 * (a + 1))
    print("failed", p, file=sys.stderr)


fs = sorted(glob.glob(os.path.join(sys.argv[1], "*", "output.wav")))
with cf.ThreadPoolExecutor(10) as ex:
    list(ex.map(run, fs))
print("transcribed", len(fs))
