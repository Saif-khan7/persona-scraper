#!/usr/bin/env python
"""
Build a persona Markdown file from Reddit SQLite rows using Gemini.
------------------------------------------------------------------
python persona_builder.py <username> --db reddit_cache.db
python persona_builder.py <username> --db reddit_cache.db --no-thinking
"""

import argparse
import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from google import genai
from google.genai import types
from google.api_core.exceptions import NotFound

# ─────────────────────────  CONFIG  ─────────────────────────
MAX_CHARS = 3_500
DEFAULT_MODEL = "gemini-2.5-flash"   # will fall back to gemini-pro if 404

SYSTEM_PROMPT = """You are a senior CX strategist creating an executive‑style
marketing persona. Synthesize the EVIDENCE provided, citing each fact like [C3].
Return Markdown with:

1. Demographics
2. Professional Background
3. Interests & Motivations
4. Communication Style
5. Pain Points / Needs
6. Representative Quote (one line)
"""

# ─────────────────────────  HELPERS  ────────────────────────
def chunk(text: str, max_len: int = MAX_CHARS) -> list[str]:
    words, buf, out = text.split(), [], []
    for w in words:
        if sum(len(x) + 1 for x in buf) + len(w) > max_len:
            out.append(" ".join(buf)); buf = []
        buf.append(w)
    if buf: out.append(" ".join(buf))
    return out


def clean(txt: str):
    txt = re.sub(r"https?://\S+", "", txt)
    txt = re.sub(r"&gt;.*", "", txt)
    return txt.strip()


def load_corpus(db: Path, user: str) -> str:
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT title, body FROM reddit_items WHERE target_user=?", (user,))
    rows = cur.fetchall(); con.close()
    return clean("\n".join(" ".join(map(str, r)) for r in rows if any(r)))


def gen_content(client, model: str, prompt: str, no_think: bool):
    cfg = None
    if no_think and model.startswith("gemini-2.5"):
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=cfg,
    ).text.strip()


# ─────────────────────────  MAIN  ──────────────────────────
def build_persona(user: str, db: Path, out: Path, no_thinking: bool):
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Add GEMINI_API_KEY to your .env file")

    client = genai.Client()  # key is read from env

    # choose model (env var override possible)
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    print(f"🔄  Attempting model: {model}"
          + ("  (thinking disabled)" if no_thinking else ""))

    raw = load_corpus(db, user)
    if not raw:
        raise RuntimeError(f"No cached posts for '{user}' in {db}")

    # summarise each chunk
    chunks = chunk(raw)
    snippets = []
    for i, c in enumerate(tqdm(chunks, desc="Gemini summaries")):
        try:
            snippets.append(
                f"[C{i}] {gen_content(client, model, f'### EVIDENCE CHUNK [{i}]\n{c}', no_thinking)}"
            )
        except NotFound:
            # auto‑fallback once
            if model != "gemini-pro":
                print("⚠️  Model not found; falling back to gemini-pro.")
                model = "gemini-pro"
                snippets.append(
                    f"[C{i}] {gen_content(client, model, f'### EVIDENCE CHUNK [{i}]\n{c}', False)}"
                )
            else:
                raise

    # final persona generation
    final_prompt = f"{SYSTEM_PROMPT}\n\n### FULL EVIDENCE\n" + "\n".join(snippets)
    persona_md = gen_content(client, model, final_prompt, no_thinking)

    out.parent.mkdir(exist_ok=True)
    out.write_text(persona_md, encoding="utf-8")
    print(f"✅  Persona written to {out}")


# ─────────────────────────  CLI  ───────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a persona via Gemini API")
    p.add_argument("username")
    p.add_argument("--db", default="reddit_cache.db")
    p.add_argument("--outfile", default=None)
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable 'thinking' (Gemini‑2.5 only) for speed")
    args = p.parse_args()

    output_path = Path(args.outfile or f"output/{args.username}_persona.txt")
    build_persona(args.username, Path(args.db), output_path, args.no_thinking)
