import argparse, os, sys, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True)
ap.add_argument("--outputs", required=True)
ap.add_argument("--split", default="all")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
ids = (
    sorted(os.listdir(a.audio))
    if a.split == "all"
    else open(os.path.join(here, "splits", f"{a.split}.txt")).read().split()
)
problems = []
for i in ids:
    p = os.path.join(a.outputs, i, "output.wav")
    if not os.path.exists(p):
        problems.append(f"{i}: missing")
        continue
    o, n = sf.info(p), sf.info(os.path.join(a.audio, i, "input.wav"))
    if o.channels != 1:
        problems.append(f"{i}: {o.channels} channels (need mono)")
    if o.samplerate < 16000:
        problems.append(f"{i}: {o.samplerate} Hz (need >= 16 kHz)")
    if abs(o.duration - n.duration) > 0.1:
        problems.append(f"{i}: {o.duration:.2f} s vs input {n.duration:.2f} s")
print(f"{len(ids)} items checked, {len(problems)} problems")
for x in problems[:50]:
    print(" ", x)
sys.exit(1 if problems else 0)
