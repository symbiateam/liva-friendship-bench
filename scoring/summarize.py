import argparse, collections, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MOMENTS, load_items

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ratings", default="ratings.jsonl")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
items = load_items(a.data)
S = collections.defaultdict(lambda: collections.defaultdict(list))
J = collections.defaultdict(dict)
for x in map(json.loads, open(a.ratings)):
    J[(x["system"], x["item"])][x["judge"]] = x["score"]
for (sysname, i), js in J.items():
    if len(js) == 2:
        S[sysname][i] = list(js.values())


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def stat(per_item, ids):
    v = [mean(per_item[i]) for i in ids if i in per_item]
    return mean(v), len(v)


def ci(per_item, ids):
    conv = collections.defaultdict(list)
    for i in ids:
        if i in per_item:
            conv[items[i]["conversation"]].append(mean(per_item[i]))
    cs = list(conv)
    rng = random.Random(0)
    bs = []
    for _ in range(a.boot):
        v = [s for c in (rng.choice(cs) for _ in cs) for s in conv[c]]
        bs.append(mean(v))
    bs.sort()
    return bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


print("| System | Overall (95 % CI) | " + " | ".join(MOMENTS) + " |")
print("|---|---|" + "---|" * len(MOMENTS))
for sysname, per_item in sorted(S.items()):
    ids = list(per_item)
    o, n = stat(per_item, ids)
    lo, hi = ci(per_item, ids)
    row = [f'{stat(per_item, [i for i in ids if items[i]["moment"] == m])[0]:.2f}' for m in MOMENTS]
    print(f"| {sysname} (n={n}) | {o:.2f} ({lo:.2f}-{hi:.2f}) | " + " | ".join(row) + " |")
