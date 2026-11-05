import argparse, os, numpy as np, soundfile as sf

INSTRUCTIONS = "You are a friend in a conversation. ALWAYS respond in the SAME language as the user."
SR = 24000
FRAME = SR // 50


class MySystem:
    def __init__(self, instructions: str): ...
    def step(self, frame: np.ndarray) -> np.ndarray:
        return np.zeros_like(frame)


def run(item_dir, out_dir):
    x, sr = sf.read(os.path.join(item_dir, "input.wav"), dtype="float32", always_2d=True)
    x = x.mean(1)
    assert sr == SR, "resample to 24 kHz first (or change SR)"
    sys_ = MySystem(INSTRUCTIONS)
    out = np.zeros_like(x)
    for i in range(0, len(x), FRAME):
        f = x[i : i + FRAME]
        out[i : i + len(f)] = sys_.step(f)[: len(f)]
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "output.wav"), out, SR, subtype="PCM_16")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for i in sorted(os.listdir(a.items)):
        if not os.path.exists(os.path.join(a.out, i, "output.wav")):
            run(os.path.join(a.items, i), os.path.join(a.out, i))
