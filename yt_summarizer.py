#!/usr/bin/env python3
import os
import re
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from urllib.parse import urlparse, parse_qs

import markdown

from rich.console import Console
from rich.markdown import Markdown

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
)
try:
    from youtube_transcript_api._errors import TooManyRequests
except Exception:
    class TooManyRequests(Exception):
        pass

try:
    from openai import OpenAI
except Exception:
    import openai
    OpenAI = None

console = Console()

# ---------- Helpers ----------
def parse_video_id(video_or_url: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", video_or_url):
        return video_or_url
    try:
        u = urlparse(video_or_url)
        if u.netloc in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
            qs = parse_qs(u.query)
            vid = qs.get("v", [None])[0]
            if vid:
                return vid
        if u.netloc == "youtu.be":
            vid = u.path.strip("/").split("/")[0]
            if vid:
                return vid
    except Exception:
        pass
    raise ValueError(f"Could not parse a valid YouTube video id from: {video_or_url}")

def cache_path_for(video_id: str, cache_dir: Path | None) -> Path:
    base = Path(".") if cache_dir is None else cache_dir
    return base / f"{video_id}.json"

def load_cached_transcript(video_id: str, cache_dir: Path | None) -> dict | None:
    p = cache_path_for(video_id, cache_dir)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cached_transcript(video_id: str, languages: List[str], raw_captions: list, cache_dir: Path | None):
    p = cache_path_for(video_id, cache_dir)
    payload = {
        "video_id": video_id,
        "languages": languages,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "youtube-transcript-api",
        "captions": raw_captions,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def fetch_transcript(video_id: str, languages: List[str], preserve_formatting: bool) -> list[dict]:
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id, languages=languages, preserve_formatting=preserve_formatting)
    return fetched.to_raw_data()

def captions_to_text(captions) -> str:
    return " ".join(c["text"].strip() for c in captions).strip()

def markdown_to_html(md: str) -> str:
    """Convert Markdown text into full HTML document."""
    md = normalize_list_indentation(md)
    body = markdown.markdown(
        md,
        extensions=[
            # Core / built-in
            "fenced_code",
            "tables",
            "abbr",
            "attr_list",
            "def_list",
            "footnotes",
            "admonition",
            "codehilite",
            "nl2br",
            "sane_lists",
            "smarty",
            "toc",
            "meta",

            # Pymdown Extensions
            "pymdownx.superfences",
            "pymdownx.highlight",
            "pymdownx.magiclink",
            "pymdownx.emoji",
            "pymdownx.tasklist",
            "pymdownx.details",
            "pymdownx.arithmatex",  # for math
        ]
    )
    return f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n<link rel=\"stylesheet\" href=\"style.css\">\n</head>\n<body>\n{body}\n</body>\n</html>"

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    if OpenAI is not None:
        return OpenAI()
    openai.api_key = api_key
    return None

def chunk_text(text: str, max_chars: int) -> List[str]:
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(n, i + max_chars)
        cut = text.rfind(".", i, end)
        cut = end if cut == -1 else cut + 1
        chunks.append(text[i:cut].strip())
        i = cut
    return [c for c in chunks if c]

def _single_call_summary(client, model: str, text: str, is_final: bool, lang: str) -> str:
    sys = (
        "Your goal is to generate a Markdown summary based on YouTube transcript."
        "Use headings (#, ##, ###), bullet points(*), and any other markdown."
        "Keep it faithful to the transcript; do not invent facts. "
    )
    sys += f"Generate the summary in this language: {lang}"

    prompt = (
        ("FINAL SYNTHESIS. " if is_final else "")
        + "Summarize the following YouTube talk transcript into Markdown.\n\n"
        f"=== TRANSCRIPT START ===\n{text}\n=== TRANSCRIPT END ==="
    )
    if OpenAI is not None:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    else:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}],
        )
        return resp["choices"][0]["message"]["content"].strip()

def summarize_markdown(transcript_text: str, model: str, lang: str) -> str:
    client = get_openai_client()
    MAX_CHARS = 50000  # fixed

    if len(transcript_text) <= MAX_CHARS:
        return _single_call_summary(client, model, transcript_text, False, lang)

    chunks = chunk_text(transcript_text, MAX_CHARS)
    partial_summaries = []
    for idx, ch in enumerate(chunks, 1):
        console.log(f"[grey62]Summarizing chunk {idx}/{len(chunks)} ({len(ch)} chars)[/grey62]")
        partial_summaries.append(_single_call_summary(client, model, ch, False, lang))
    combined = "\n\n".join(partial_summaries)
    console.log("[grey62]Synthesizing final summary from chunk summaries…[/grey62]")
    raw_summary = _single_call_summary(client, model, combined, True, lang)
    
    return enrich_markdown(raw_summary, client, model)


def normalize_list_indentation(md: str) -> str:
    """
    Replace 2-space indents at the start of list items with 4 spaces.
    Only affects nested list indentation, not other content.
    """
    # Matches lines that start with 2 spaces followed by a list marker
    return re.sub(r'^(  +)([-*+])', lambda m: " " * (len(m.group(1)) * 2) + m.group(2), md, flags=re.MULTILINE)

def enrich_markdown(md, client, model):
    sys = (
        "Enrich this markdown by using heading, table, or any other component. "
        "Use bold (**) for the proper name like Tools name, speaker name, company name. "
    )
    prompt = (
        f"=== TEXT START ===\n{md}\n=== TEXT END ==="
    )
    if OpenAI is not None:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    else:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}],
        )
        return resp["choices"][0]["message"]["content"].strip()


# ---------- Orchestration ----------
def run(video, languages, model):
    video_id = parse_video_id(video)
    captions = (load_cached_transcript(video_id, None) or {}).get("captions")

    if not captions:
        try:
            captions = fetch_transcript(video_id, languages, preserve_formatting=False)
        except (NoTranscriptFound, TranscriptsDisabled):
            console.print("[red]No transcript available.[/red]"); return
        except VideoUnavailable:
            console.print("[red]Video is unavailable.[/red]"); return
        except TooManyRequests:
            console.print("[red]Rate limited by YouTube.[/red]"); return
        except Exception as e:
            console.print(f"[red]Failed to fetch transcript: {e}[/red]"); return
        save_cached_transcript(video_id, languages, captions, None)

    transcript_text = captions_to_text(captions)
    console.print("\n[bold]Transcript (first 600 chars):[/bold]")
    console.print(transcript_text[:600] + ("…" if len(transcript_text) > 600 else ""))
    console.print()

    summary_md = summarize_markdown(transcript_text, model=model, lang=languages[0])

    # Save outputs
    md_path = f"{video_id}.md"
    html_path = f"{video_id}.html"
    html_str=markdown_to_html(summary_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    console.print(f"[green]Saved Markdown:[/green] {md_path}")
    console.print(f"[green]Saved HTML:[/green] {html_path}")

    console.print("\n[bold green]=== Summary (Markdown) ===[/bold green]\n")
    console.print(Markdown(summary_md), soft_wrap=True)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube captions and summarize to Markdown/HTML.")
    parser.add_argument("video", help="YouTube video ID or URL")
    parser.add_argument("-l", "--languages", nargs="+", default=["en"], help="Language preference list (default: en).")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI model (default: gpt-5-nano)")
    args = parser.parse_args()

    run(
        video=args.video,
        languages=args.languages,
        model=args.model,
    )

if __name__ == "__main__":
    main()
