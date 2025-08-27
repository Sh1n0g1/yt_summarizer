# YouTube → Markdown/HTML Summarizer

Fetch a YouTube video’s transcript (no downloading audio/video required) and summarize it into **clean Markdown** and a matching **HTML** file using OpenAI. Renders a preview of the summary in the terminal via `rich`.

- **Input**: YouTube video ID or URL  
- **Output**: `<video_id>.md` and `<video_id>.html`  
- **Cache**: Stores raw captions as JSON so you don’t hit YouTube repeatedly

---

## Quick Start

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 3) Set your OpenAI API key
# Linux
export OPENAI_API_KEY="sk-..."
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# 4) Run the tool (example video id)
python yt_summarizer.py 6idT7hsb7QE
```

On success you’ll see:

- `6idT7hsb7QE.md`
- `6idT7hsb7QE.html`

Plus a formatted Markdown preview in your terminal.

---

## Command-line Usage

```
usage: yt_summarizer.py [-h] [-l LANGUAGES [LANGUAGES ...]] [--preserve-formatting]
                        [--pause-threshold PAUSE_THRESHOLD] [--flat]
                        [--model MODEL] [--cache-dir CACHE_DIR] [--force-refresh]
                        video

Fetch YouTube captions and summarize to Markdown/HTML.

positional arguments:
  video                 YouTube video ID or URL

options:
  -h, --help            show this help message and exit
  -l, --languages       Language preference list (default: ["en"])
  --model               OpenAI model to use (default: gpt-5-nano)
```

### What the flags do

- **`-l/--languages`**: List of language codes to attempt, in order. First element also controls summary language if it’s Japanese (e.g., `ja`).
- **`--model`**: Any OpenAI chat model name you have access to.

---

## Examples

### 1) Minimal (ID)
```bash
python yt_summarizer.py 6idT7hsb7QE
```

### 2) With full URL
```bash
python yt_summarizer.py "https://www.youtube.com/watch?v=6idT7hsb7QE"
```

### 3) Japanese summary preference (try JA first, then EN)
```bash
python yt_summarizer.py 6idT7hsb7QE -l ja en --model gpt-5
```

---

## How It Works

1. **Video ID parsing**  
   Accepts plain IDs like `6idT7hsb7QE` or standard YouTube URLs (including `youtu.be` links).

2. **Transcript retrieval**  
   Uses [`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/) to fetch captions (no browser automation). Results are cached as `<video_id>.json`.

3. **Text assembly**  
   - *Flat mode*: joins captions into one block.  
   - *Pause-aware mode*: if `--pause-threshold > 0`, inserts paragraph breaks on long silences.

4. **Summarization**  
   Sends the assembled text to OpenAI Chat Completions with a “precise note-taker” system prompt.  
   - If text length exceeds an internal threshold, it’s chunked and synthesized into a final summary.

5. **Output**  
   - Saves **Markdown** to `<video_id>.md`.  
   - Converts that Markdown to a standalone **HTML** file `<video_id>.html`.

---

## Requirements

- Python 3.9+ (recommended)
- Packages: `youtube-transcript-api`, `rich`, `markdown`, `openai`
- Environment: `OPENAI_API_KEY` must be set

Install:
```bash
pip install youtube-transcript-api rich markdown openai
```

---

## Troubleshooting

- **`OPENAI_API_KEY not set in environment.`**  
  Set it before running:
  - macOS/Linux: `export OPENAI_API_KEY="sk-..."`
  - Windows PowerShell: `$env:OPENAI_API_KEY="sk-..."`

- **`No transcript available.`**  
  Some videos have no captions or they’re disabled. Try another language order:
  ```bash
  python yt_summarizer.py <id> -l en ja es
  ```

- **`Video is unavailable.`**  
  The video may be private/removed/region-blocked.

- **`Rate limited by YouTube.`**  
  `youtube-transcript-api` can hit rate limits. Wait a bit or use cached results (omit `--force-refresh`).

- **Model errors or quota issues**  
  Ensure the model name is valid for your account and you have sufficient quota.

---

## Notes

- The first language in `-l/--languages` also controls the output language **only when it starts with `ja`** (Japanese). Otherwise summaries default to English.
- The default model is `gpt-5-nano`. You can switch via `--model`.

---

