import argparse, concurrent.futures as cf, json, os, re, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUBRIC, WINDOW, JUDGES, load_items

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--outputs")
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="public")
ap.add_argument("--out", default="ratings.jsonl")
a = ap.parse_args()
OK = os.environ.get("OPENAI_API_KEY")
GK = os.environ.get("GEMINI_API_KEY")
if not (OK and GK):
    sys.exit("Set OPENAI_API_KEY and GEMINI_API_KEY")
items = load_items(a.data)
ids = [
    i
    for i in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits", f"{a.split}.txt"))
    .read()
    .split()
    if i in items
]


def reply(i):
    if a.outputs is None:
        return items[i].get("real_listener_text")
    p = os.path.join(a.outputs, i, "output_scribe.json")
    if not os.path.exists(p):
        return None
    off = items[i]["turn_end_s"]
    return " ".join(
        w["w"] for w in json.load(open(p)) if w.get("s") is not None and off + WINDOW[0] <= w["s"] <= off + WINDOW[1]
    ).strip()


def post(url, body, hdr):
    for k in range(6):
        try:
            return json.loads(
                urllib.request.urlopen(
                    urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr), timeout=90
                ).read()
            )
        except Exception:
            time.sleep(3 * (k + 1))


def judge(name, msg):
    if name == "gpt-4.1":
        o = post(
            "https://api.openai.com/v1/chat/completions",
            dict(
                model="gpt-4.1",
                temperature=0,
                response_format=dict(type="json_object"),
                messages=[dict(role="system", content=RUBRIC), dict(role="user", content=msg)],
            ),
            {"Authorization": "Bearer " + OK, "Content-Type": "application/json"},
        )
        t = o["choices"][0]["message"]["content"] if o else ""
    else:
        o = post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={GK}",
            dict(
                system_instruction=dict(parts=[dict(text=RUBRIC)]),
                contents=[dict(role="user", parts=[dict(text=msg)])],
                generationConfig=dict(temperature=0, maxOutputTokens=800, responseMimeType="application/json"),
            ),
            {"Content-Type": "application/json"},
        )
        try:
            t = o["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            t = ""
    m = re.search(r'"score"\s*:\s*([1-5])', t or "")
    return int(m.group(1)) if m else None


done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        x = json.loads(l)
        done.add((x["item"], x["system"], x["judge"]))
lock = threading.Lock()
fo = open(a.out, "a")


def one(job):
    i, j = job
    if (i, a.name, j) in done:
        return
    text = reply(i)
    if text is None:
        return
    s = judge(j, f"SPEAKER: {items[i]['speaker_text']}\nLISTENER: {text or '(stays silent)'}")
    if s is None:
        return
    with lock:
        fo.write(json.dumps(dict(item=i, system=a.name, judge=j, score=s, reply=text)) + "\n")
        fo.flush()


with cf.ThreadPoolExecutor(16) as ex:
    list(ex.map(one, [(i, j) for i in ids for j in JUDGES]))
print("rated", a.name, len(ids), "items ->", a.out)
