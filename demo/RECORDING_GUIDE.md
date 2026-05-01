# Recording Guide — agent-recall-ai Demo Video

## Setup (2 min)

1. Open Chrome → navigate to `http://localhost:7788`
2. Press **F11** to go fullscreen (or set window to exactly 1280×720)
3. Open your screen recorder (OBS, Loom, or built-in):
   - **Windows**: Win+G (Xbox Game Bar) or OBS
   - **Mac**: Cmd+Shift+5
   - **Loom**: browser extension, click record
4. Set audio input to your microphone
5. Set recording resolution to **1280×720**
6. Hit **record**, then press **Space / → arrow** to advance slides

The deck auto-advances. To control manually: **← →** arrow keys or Space.

---

## Voiceover Script

### Slide 1 — Hook (5 sec)
> "Your AI agent just died."
> *(pause)*
> "Three hours into a complex task — context window full, session terminated, everything gone."

---

### Slide 2 — Problem visualization (7 sec)
> "This is what happens on every long-running agent task. The context window fills up — and when it hits the limit, the entire session is gone. No recovery. No memory. Start from scratch."

---

### Slide 3 — Introducing agent-recall-ai (5 sec)
> "Introducing **agent-recall-ai** — framework-agnostic checkpoints for AI agents. Works with OpenAI, Anthropic, LangChain, LangGraph, CrewAI — any framework."

---

### Slide 4 — Code: saving a checkpoint (7 sec)
> "It's one context manager. Wrap your agent task with `Checkpoint`, and you get automatic persistence — goals, decisions, files touched, token usage — all saved on exit, even on a crash."

---

### Slide 5 — What gets saved (6 sec)
> "A checkpoint captures everything that matters: goals and constraints, the full decision log with reasoning, every file touched, real-time cost tracking, monitor alerts, and exact next steps for resuming."

---

### Slide 6 — Resume (6 sec)
> "When the context window fills, just call `resume`. You get back a ready-to-use prompt string — goals, constraints, decisions, next steps — everything the new session needs to pick up exactly where the old one stopped."

---

### Slide 7 — Monitors (7 sec)
> "Built-in monitors protect your session in real time. Budget alerts before you blow past your cost limit. Context pressure warnings at 75 and 90 percent. Drift detection when a decision violates a constraint. And auto-compression when tool outputs bloat your context."

---

### Slide 8 — Stats (6 sec)
> "Six supported frameworks. Fourteen PII redaction categories so secrets never hit your logs. 354 tests passing. And zero config required — SQLite out of the box, Redis for production, OpenTelemetry for observability."

---

### Slide 9 — CTA (8 sec)
> "Your agent never starts over again."
> *(pause)*
> "`pip install agent-recall-ai`"
> "Open source, MIT licensed, on GitHub right now. Star the repo, drop it into your next agent, and let us know what you build."

---

## Cuts

| Platform | Length  | Slides to include         |
|----------|---------|---------------------------|
| X / Twitter | 60s  | 1, 2, 3, 4, 6, 9          |
| LinkedIn    | 90s  | All 9                     |
| YouTube     | 90s  | All 9 + add intro card    |

---

## Post-production tips

- Add background music (lo-fi / ambient) at ~15% volume
- Trim any pauses between slides
- Export at **1280×720 H.264**, 30fps
- For X: keep under 140 seconds, square crop (1080×1080) also works
- Add captions using auto-caption in CapCut, DaVinci Resolve, or YouTube Studio
