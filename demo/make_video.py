"""
agent-recall-ai demo video generator
─────────────────────────────────────
1. Generates per-slide MP3 voiceovers via Microsoft Edge TTS (free, no key needed)
2. Records each slide as a video clip using Playwright + Chromium
3. Stitches everything together with ffmpeg

Output: demo/agent-recall-ai-demo.mp4  (1280×720, H.264)
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ── config ────────────────────────────────────────────────────────────────────
DEMO_URL   = "http://localhost:7799"
OUT_DIR    = Path(__file__).parent / "render"
FINAL_OUT  = Path(__file__).parent / "agent-recall-ai-demo.mp4"
FFMPEG     = r"C:\Users\srina\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
VOICE      = "en-US-AndrewMultilingualNeural"  # clear, natural male voice
WIDTH, HEIGHT = 1280, 720

# ── per-slide script + hold duration (seconds) ───────────────────────────────
# Slides match the new 3-act story: Problem -> Fix -> Power
SLIDES: list[dict] = [
    {
        "idx": 0,
        "hold": 3.8,   # hold for the full crash animation to play out
        "script": (
            "3 hours of work. One context limit. "
            "Watch what happens. "
            "The agent is running fine — context at 12%, 35%, 68%. "
            "Then 78% — a warning. 96% — critical. "
            "And then: Context Window Exceeded. "
            "Process terminated. Session state: not saved. "
            "3 hours of work. 87 migrations. Every decision. Gone."
        ),
    },
    {
        "idx": 1,
        "hold": 2.0,
        "script": (
            "Let's look at exactly what just disappeared. "
            "87 migration files — gone. "
            "Every architectural decision with its reasoning — gone. "
            "4 dollars and 80 cents in API costs — completely wasted. "
            "3 hours of irreplaceable progress. "
            "This happens to every AI engineer running long tasks. "
            "There has to be a better way."
        ),
    },
    {
        "idx": 2,
        "hold": 3.5,   # hold for parallel terminal animations
        "script": (
            "Same task. One change. Completely different outcome. "
            "On the left — no protection. The agent works, hits the limit, crashes. State not saved. Start over. "
            "On the right — with agent-recall-ai. "
            "One context manager wrapping the same code. "
            "The agent hits the same limit — but this time: checkpoint saved, state preserved. "
            "Resume anytime. Zero work lost."
        ),
    },
    {
        "idx": 3,
        "hold": 2.0,   # hold for card stagger animation
        "script": (
            "A checkpoint automatically captures everything that matters. "
            "Goals and constraints — what the agent is trying to do and what lines it cannot cross. "
            "The full decision log with reasoning and alternatives rejected. "
            "Every file touched. Real-time cost tracking. "
            "Monitor alerts. And exact next steps for resuming."
        ),
    },
    {
        "idx": 4,
        "hold": 2.5,   # hold for tick animations
        "script": (
            "When the context window fills, just call resume. "
            "One line. You get back a complete prompt string. "
            "The new session knows the goal, the constraints, every decision made so far, "
            "and exactly which step to continue from. "
            "Goal remembered. Decisions carried over. Picks up at step 88. "
            "Cost saved: zero dollars wasted — versus 4 dollars 80 starting over."
        ),
    },
    {
        "idx": 5,
        "hold": 3.0,   # hold for alert sequence
        "script": (
            "Built-in monitors protect every session in real time. "
            "Add them in one line. "
            "Cost monitor — alerts before you blow past your budget. "
            "Token monitor — warns at 75 and 90 percent context usage. "
            "Drift monitor — catches when a decision violates a constraint. "
            "Tool bloat monitor — auto-compresses oversized tool outputs. "
            "And every alert is saved with the checkpoint."
        ),
    },
    {
        "idx": 6,
        "hold": 2.0,   # hold for card stagger
        "script": (
            "It works with every major framework. "
            "OpenAI, Anthropic, LangChain, LangGraph, CrewAI — "
            "each has a one-line adapter. "
            "Or use the Checkpoint class directly with any framework at all."
        ),
    },
    {
        "idx": 7,
        "hold": 2.5,   # hold for stat animations
        "script": (
            "Six frameworks supported. "
            "Fourteen P-I-I redaction categories — secrets never hit your logs. "
            "354 tests passing. "
            "Zero config required. "
            "SQLite out of the box, Redis for production, "
            "OpenTelemetry and Datadog for observability."
        ),
    },
    {
        "idx": 8,
        "hold": 2.5,
        "script": (
            "Your agent never starts over again. "
            "pip install agent-recall-ai. "
            "Open source, MIT licensed, on GitHub right now. "
            "Star the repo, drop it into your next agent, and let us know what you build."
        ),
    },
]


# ── step 1: generate voiceovers ───────────────────────────────────────────────
async def gen_voiceovers() -> list[Path]:
    import edge_tts

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for s in SLIDES:
        out = OUT_DIR / f"voice_{s['idx']:02d}.mp3"
        if out.exists():
            print(f"  [voice] slide {s['idx']+1} — cached")
        else:
            print(f"  [voice] slide {s['idx']+1} — generating …")
            comm = edge_tts.Communicate(s["script"], VOICE, rate="+5%")
            await comm.save(str(out))
            print(f"          saved -> {out.name}")
        paths.append(out)

    return paths


# ── step 2: get audio duration ────────────────────────────────────────────────
def audio_duration(mp3: Path) -> float:
    """Return duration of an MP3 in seconds using ffprobe."""
    result = subprocess.run(
        [
            FFMPEG.replace("ffmpeg.exe", "ffprobe.exe"),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(mp3),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 4.0


# ── step 3: record slide clips ────────────────────────────────────────────────
async def record_slides(voice_paths: list[Path]) -> list[Path]:
    from playwright.async_api import async_playwright

    clips: list[Path] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                f"--window-size={WIDTH},{HEIGHT}",
                "--force-device-scale-factor=1",
                "--disable-scrollbars",
                "--hide-scrollbars",
                "--font-render-hinting=none",
            ],
        )

        for s, vpath in zip(SLIDES, voice_paths):
            idx      = s["idx"]
            hold     = s["hold"]
            vo_dur   = audio_duration(vpath)
            clip_dur = vo_dur + hold
            clip_out = OUT_DIR / f"clip_{idx:02d}.webm"

            if clip_out.exists():
                print(f"  [clip]  slide {idx+1} — cached ({clip_dur:.1f}s)")
                clips.append(clip_out)
                continue

            print(f"  [clip]  slide {idx+1} — recording {clip_dur:.1f}s ...")

            context = await browser.new_context(
                viewport={"width": WIDTH, "height": HEIGHT},
                device_scale_factor=1,
                record_video_dir=str(OUT_DIR / f"_raw_{idx:02d}"),
                record_video_size={"width": WIDTH, "height": HEIGHT},
            )
            page = await context.new_page()
            # Inject CSS to guarantee no scrollbars and exact fill
            await page.add_init_script("""
                document.addEventListener('DOMContentLoaded', () => {
                    document.documentElement.style.overflow = 'hidden';
                    document.body.style.overflow = 'hidden';
                });
            """)
            await page.goto(DEMO_URL)
            await page.wait_for_load_state("networkidle")
            # Wait for Google Fonts to load
            await page.wait_for_timeout(1200)

            # Disable auto-advance and jump to correct slide
            await page.evaluate("clearTimeout(timer)")
            await page.evaluate(f"go({idx})")
            await asyncio.sleep(0.4)  # let slide transition settle

            # Record the slide for its full duration
            await asyncio.sleep(clip_dur)

            await context.close()  # flushes video to disk

            # playwright saves video with a random name — find and rename it
            raw_dir = OUT_DIR / f"_raw_{idx:02d}"
            webm_files = list(raw_dir.glob("*.webm"))
            if webm_files:
                webm_files[0].rename(clip_out)
                shutil.rmtree(raw_dir, ignore_errors=True)
                print(f"          saved → {clip_out.name}")
            else:
                print(f"  [WARN]  no video found for slide {idx+1}")

            clips.append(clip_out)

        await browser.close()

    return clips


# ── step 4: mux audio + video per clip, then concatenate ─────────────────────
def build_final_video(clips: list[Path], voices: list[Path]) -> Path:
    muxed: list[Path] = []

    print("\n[mux] combining audio + video per slide …")
    for i, (clip, voice) in enumerate(zip(clips, voices)):
        out = OUT_DIR / f"muxed_{i:02d}.mp4"
        if out.exists():
            print(f"  slide {i+1} — cached")
            muxed.append(out)
            continue

        vo_dur = audio_duration(voice)
        hold   = SLIDES[i]["hold"]
        total  = vo_dur + hold

        # pad audio with silence to match video duration
        subprocess.run([
            FFMPEG, "-y",
            "-i", str(clip),
            "-i", str(voice),
            # audio: pad end with silence to match clip length
            "-filter_complex",
            f"[1:a]apad=whole_dur={total}[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(total),
            str(out),
        ], check=True, capture_output=True)
        print(f"  slide {i+1} -> {out.name}")
        muxed.append(out)

    # write concat list
    concat_list = OUT_DIR / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in muxed)
    )

    print("\n[concat] stitching slides into final video …")
    subprocess.run([
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(FINAL_OUT),
    ], check=True, capture_output=True)

    return FINAL_OUT


# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  agent-recall-ai demo video generator")
    print("=" * 60)

    print("\n[1/3] Generating voiceovers (Microsoft Edge TTS) …")
    voices = await gen_voiceovers()

    print("\n[2/3] Recording animated slide clips (Playwright) …")
    clips = await record_slides(voices)

    print("\n[3/3] Muxing and concatenating with ffmpeg …")
    final = build_final_video(clips, voices)

    print(f"\n{'=' * 60}")
    print(f"  Done!  ->  {final}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
