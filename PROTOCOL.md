# Protocol

Every system is run the same way, so its reply can be compared with what a real person said at the same moment.

1. **Input.** For each item, stream `input.wav` to the system **in real time** (as if from a microphone), from the first sample
   to the last. The file is the speaker's own track: up to 2 s before the turn, the turn, then 6 s of silence.
2. **Instructions.** Use exactly this system prompt (or your system's closest equivalent) and nothing else, no extra context:
   `You are a friend in a conversation. ALWAYS respond in the SAME language as the user.`
3. **One session per item.** Start a fresh session for every item; no memory across items.
4. **Turn-taking.** Let the system decide when to speak (its own voice-activity or full-duplex behaviour). Do not force a
   response at the end of the turn. If the speaker talks over the system, the system's speech stops where it would stop in a
   live call.
5. **Recording.** Record what a listener would hear, on the same timeline as the input: silence while the system is quiet,
   its speech where it would be heard. Save `output.wav`: mono, at least 16 kHz (24 kHz recommended), **same duration as the
   input** (within 0.1 s). Do not trim, shift or edit the audio.
6. **Layout.** `<outputs>/<item_id>/output.wav` for every item.
7. **What is scored.** Words that start between 0.3 s before and 6 s after the end of the speaker's turn
   (`turn_end_s` in `items.jsonl`). Staying silent is a valid reply and is rated like any other.

Report the exact model id, voice and any non-default settings with your results.
