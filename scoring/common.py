import json, os

MOMENTS = ["affection", "tease", "question", "flirt", "vulnerable", "self_deprecation", "punchline"]
WINDOW = (-0.3, 6.0)
JUDGES = ["gpt-4.1", "gemini-3.8-flash"]
RUBRIC = """You rate one listener reaction in a casual voice chat between close friends or romantic partners. You see only what the SPEAKER just said and the LISTENER's reaction (bracketed tags are non-speech sounds; "(stays silent)" means no reaction within 6 seconds).
Rate how natural and fitting the reaction is for a close friend or partner at this moment, using only the speaker's line as context:
5 = exactly what a close friend or partner might naturally say here
4 = natural and fitting
3 = acceptable but generic, or slightly off
2 = awkward, off-topic, overly formal or assistant-like
1 = nonsensical or clearly inappropriate
Short replies, backchannels, laughter and silence can deserve a 5. Do not reward length, politeness or enthusiasm for their own sake.
Return JSON {"score": 1-5}."""


def load_items(data_dir):
    return {x["id"]: x for x in map(json.loads, open(os.path.join(data_dir, "items.jsonl")))}
