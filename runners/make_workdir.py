import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--split", default="all")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
ids = (
    sorted(os.listdir(a.audio))
    if a.split == "all"
    else open(os.path.join(here, "..", "splits", f"{a.split}.txt")).read().split()
)
for i in ids:
    d = os.path.join(a.work, "listener", i)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "input.wav")
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(os.path.join(a.audio, i, "input.wav")), dst)
print(len(ids), "items linked under", os.path.join(a.work, "listener"))
