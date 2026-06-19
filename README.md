# Vietnamese Voiceover (Thuyết minh) Toolkit

Turn Vietnamese subtitles into a synthesized **thuyết minh** voiceover with
[VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), keep it roughly in sync with
the subtitle timeline, and either export the narration or mux it back into a video
as a switchable audio track.

Two scripts:

| Script | Input | Output |
| --- | --- | --- |
| `srt_to_voiceover.py` | a `.srt` file (+ optional video) | narration audio, optionally muxed into the video |
| `mkv_voiceover.py` | an `.mkv` with an embedded Vietnamese subtitle | the same `.mkv` plus a new "Thuyết minh" audio track |

`mkv_voiceover.py` reuses the core of `srt_to_voiceover.py`, so **keep both files in
the same folder.**

---

Demo:

https://github.com/user-attachments/assets/e1f77403-139a-4762-9676-e8de132eed0f

---

## 1. Requirements

**Python:** 3.9 or newer (uses `argparse.BooleanOptionalAction`).

**Python packages** — install all three (or `python -m pip install -r requirements.txt`):

```bash
python -m pip install vieneu srt pydub numpy
```

| Package | Used for |
| --- | --- |
| `vieneu` | VieNeu-TTS Vietnamese text-to-speech engine (loaded lazily on first synth) |
| `srt` | Parsing `.srt` subtitle files |
| `pydub` | Assembling/exporting the narration audio (calls ffmpeg under the hood) |
| `numpy` | Fast timeline assembly (sums all clips into one buffer) |

Everything else the scripts use is in the Python standard library (`argparse`,
`subprocess`, `pathlib`, `json`, `re`, `unicodedata`, `hashlib`, `shutil`, `contextlib`,
`tempfile`, `time`, `os`, `sys`, `dataclasses`, `typing`, `concurrent.futures`).

**System tools** (NOT installable via pip — must be on PATH):

- `ffmpeg` and `ffprobe` — all audio/video work; `pydub` also depends on ffmpeg.
- **eSpeak NG** — phonemizer required by VieNeu (the model fails to init without it).

```bash
# macOS
brew install ffmpeg espeak

# Ubuntu / Debian
sudo apt install ffmpeg espeak-ng
```

**Notes**

- VieNeu's default backbone is **VieNeu-TTS v3 Turbo** — on CPU it runs torch-free
  via ONNX Runtime (no GPU needed, works well on Apple Silicon); on a CUDA GPU it
  uses PyTorch.
- Output audio is **48 kHz** and **watermarked by default**.
- **Licensing:** VieNeu-TTS v3 Turbo is Apache-2.0 (commercial OK).
- First run downloads the model from Hugging Face (one-time).

---

## 2. Quick start

### Preset voices

Pass one of these names to `--voice`. **Names contain spaces, so quote them**
(e.g. `--voice "Ngọc Lan"`):

| `--voice` | Gender | Style |
| --- | --- | --- |
| `Ngọc Linh` | Female | tươi sáng (**default**) |
| `Ngọc Lan` | Female | dịu dàng |
| `Mỹ Duyên` | Female | mượt mà |
| `Trúc Ly` | Female | trẻ trung |
| `Gia Bảo` | Male | mượt mà |
| `Thái Sơn` | Male | chắc khỏe |
| `Đức Trí` | Male | rõ ràng |
| `Xuân Vĩnh` | Male | vui tươi |
| `Trọng Hữu` | Male | uyên bác |
| `Bình An` | Male | điềm đạm |

You can also list them at runtime (in case the set changes):

```bash
python -c "from vieneu import Vieneu; [print(n,'-',d) for d,n in Vieneu().list_preset_voices()]"
```

### Commands

```bash
# A) SRT -> narration audio only
python srt_to_voiceover.py phim.srt --audio-out thuyetminh.mp3 --voice "Ngọc Lan"

# B) SRT + video -> narration mixed into the video
python srt_to_voiceover.py phim.srt --video phim.mp4 --voice "Ngọc Lan" --audio-out narration.wav

# C) MKV with embedded Vietnamese sub -> add a "Thuyết minh" track
python mkv_voiceover.py single 'movie.mkv' --voice "Ngọc Lan" --emotion storytelling \
    --duck-mode auto --merge-gap 34 --cache-dir .ttscache

# Real example (anime: no English track, so mix over the Japanese audio).
# First run WITHOUT --in-place to verify, then add --in-place to overwrite.
python mkv_voiceover.py single \
    '/Volumes/Data/Plex/tvshow/Attack on Titan (2013)/Season 01/Attack.on.Titan.S01E06.mkv' \
    --voice "Ngọc Lan" --emotion storytelling \
    --audio-lang jpn --duck-mode auto \
    --merge-gap 34 --cache-dir .ttscache

# Whole season, overwriting in place once you're happy with the result.
# --workers 2 synthesizes cues in parallel (~1.8x faster); needs --cache-dir.
python mkv_voiceover.py batch \
    '/Volumes/Data/Plex/tvshow/Attack on Titan (2013)/Season 01' \
    --voice "Ngọc Lan" --emotion storytelling --audio-lang jpn \
    --merge-gap 34 --cache-dir .ttscache --in-place --workers 2
```

---

## 3. How it works

1. **Parse + clean** each subtitle cue (strips emoji/markup, keeps `. , ! ?`,
   collapses newlines).
2. **Synthesize** each cue with VieNeu-TTS (optionally a preset voice or a cloned one).
3. **Fit to the timeline** — VieNeu has no native duration control, so each clip is
   time-stretched (pitch-preserving `atempo`) to fit its slot, capped by `--max-speed`;
   anything over the cap is allowed to overflow into the following gap.
4. **Assemble** all clips onto a silent track the length of the video.
5. **Mix / mux** with ffmpeg (`-c copy` for everything except the new track).

For MKV, the new track is built **inside ffmpeg** by mixing the film audio with the
narration, anchored to the film's length (`amix duration=first`). This keeps the
track full-length and seekable, and produces a real thuyết minh blend (film audio
audible underneath the narrator).

---

## 4. Audio balance / loudness

The film audio and narration are measured (EBU R128 / LUFS) and balanced automatically:

- `--auto-balance` (default **on**): measures both, then sets the narration gain so the
  narrator sits `--target-offset` LU above the film (default +3 LU).
- `--normalize` (default **on**): loudness-normalizes the final mix to `--target-lufs`
  (default −16 LUFS) so the track is never too quiet or too loud.
- `--duck-mode auto` (default): the film **only dips while the narrator speaks**
  (sidechain ducking) and returns to full volume otherwise — the natural-sounding option.
- `--duck-mode static --duck-db -9`: lowers the film by a fixed amount the whole time.

Tuning examples:

```bash
# Narrator a bit louder relative to the film
python mkv_voiceover.py single movie.mkv --target-offset 5

# Disable auto-balance and set the narration gain manually
python mkv_voiceover.py single movie.mkv --no-auto-balance --voice-db +2

# Old-style constant ducking, film lowered only 6 dB
python mkv_voiceover.py single movie.mkv --duck-mode static --duck-db -6
```

---

## 5. Performance

- The TTS step is the only heavy part; muxing is a near-instant stream copy.
- `--workers N` synthesizes cues in parallel across `N` processes (each cue is an
  independent TTS call), so it speeds up even a **single** file. Requires `--cache-dir`
  (workers hand results back through the cache). Measured ~1.8x at `--workers 2` on a
  10-core machine; each worker loads its own model (~1–2 GB RAM), and the threads per
  worker are auto-capped so they don't oversubscribe the CPU. Diminishing returns past
  the point where workers × model-RAM exhausts memory or all cores are busy.
- `--cache-dir DIR` memoizes each cue, so re-runs / retries skip already-rendered cues.
  It is **not** auto-deleted (that's what makes retries fast) and is created wherever you
  run the command. Use `--cache-dir auto` to place it next to each MKV
  (`<folder>/.ttscache`), and add `--clean-cache` to remove it once the run finishes.
  The cache is keyed by cue text + voice, so it mainly speeds up re-running the *same*
  file (e.g. after an interrupt, or when tweaking mix/balance without re-synthesizing);
  different episodes share little. One-shot clean run: `--cache-dir auto --clean-cache`.
- `--merge-gap MS` merges cues separated by small gaps to cut the number of TTS calls
  (also reads more naturally). Try `--merge-gap 34`.
- `--analyze-seconds N` limits how much film audio is scanned for loudness (default 300).
- Progress is logged as six numbered stages, each with a `done (Xs)` marker, plus a live
  `done/total` line with ETA during synthesis and a final `DONE <path> (Xs total)`. With
  `--workers`, the parallel pre-synthesis shows a slider with elapsed/ETA that advances
  per cue (`prefill [███░░░] 740/1818 (40.7%) elapsed 2:10 eta 3:09`); the ETA settles
  after the models finish loading. Noisy ffmpeg / backend init logs are suppressed (pass
  `--verbose` to see them).
- **Tip:** test on a 60-second clip first.
  ```bash
  ffmpeg -i movie.mkv -t 60 -c copy -map 0 test60.mkv
  python mkv_voiceover.py single test60.mkv --voice "Ngọc Lan" --cache-dir .ttscache
  ```

---

## 6. Command reference

### `srt_to_voiceover.py SRT [options]`

| Option | Default | Description |
| --- | --- | --- |
| `--audio-out PATH` | `narration.wav` | Narration output (`.wav`/`.mp3`) |
| `--video PATH` | – | Mux narration into this video |
| `--video-out PATH` | auto | Output video path |
| `--replace-audio` | off | Replace original audio instead of mixing over it |
| `--duck-db N` | `-12` | Original-audio level when mixing |
| `--voice ID` | – | Preset voice name, quote if it has spaces (e.g. `"Ngọc Lan"`) |
| `--emotion` | `natural` | `natural` or `storytelling` |
| `--clone-audio` / `--clone-text` | – | Clone a narrator voice from a 3–5 s clip |
| `--max-speed N` | `1.5` | Max pitch-preserving speed-up to fit a slot |
| `--speed N` | `1.0` | Global pitch-preserving narration speed (1.1–1.25 transparent) |
| `--merge-gap MS` | `0` | Merge cues with gaps ≤ this |
| `--cache-dir DIR` | – | Per-cue synthesis cache |
| `--workers N` | `1` | Parallel TTS worker processes (needs `--cache-dir`) |

### `mkv_voiceover.py single MKV [options]` / `batch FOLDER [options]`

Common options:

| Option | Default | Description |
| --- | --- | --- |
| `--voice ID` | – | Preset voice name, quote if it has spaces (e.g. `"Ngọc Lan"`) |
| `--emotion` | `natural` | `natural` or `storytelling` |
| `--clone-audio` / `--clone-text` | – | Voice cloning |
| `--sub-lang` | `vie` | Subtitle language tag to match |
| `--sub-index N` | auto | Force the subtitle stream (`0:s:N`) |
| `--audio-lang` | `eng` | Preferred non-Vietnamese mix-base language (VI always excluded) |
| `--audio-index N` | auto | Force the mix-base audio stream (`0:a:N`) |
| `--auto-balance` / `--no-auto-balance` | on | Measure loudness and auto-set voice gain |
| `--target-offset LU` | `3` | How far the narrator sits above the film |
| `--normalize` / `--no-normalize` | on | Loudness-normalize the final mix |
| `--target-lufs N` | `-16` | Integrated loudness target |
| `--voice-db N` | `0` | Manual narration gain (when auto-balance off) |
| `--duck-mode` | `auto` | `auto` (sidechain) or `static` |
| `--duck-db N` | `-9` | Film level in `static` mode |
| `--analyze-seconds N` | `300` | Film audio scanned for loudness |
| `--max-speed N` | `1.5` | Max speed-up to fit a slot |
| `--speed N` | `1.0` | Global pitch-preserving narration speed (1.1–1.25 transparent) |
| `--merge-gap MS` | `0` | Merge close cues |
| `--cache-dir DIR` | – | Per-cue synthesis cache (`auto` = next to each MKV) |
| `--clean-cache` | off | Delete the cache dir after finishing |
| `--workers N` | `1` | Parallel TTS workers per file, ~1.8x at `2` (needs `--cache-dir`; ~1–2 GB RAM each) |
| `--audio-codec` | `libopus` | Codec for the new track |
| `--bitrate` | `128k` | Bitrate for the new track |
| `--track-title` | `Thuyết minh (TTS)` | Track title metadata |
| `--set-default` | off | Make the narration the default audio track |
| `--no-mix` | off | New track = narration only (no film audio) |
| `--in-place` | off | Overwrite the source `.mkv` (atomic swap) |
| `--if-exists` | `override` | If a narration track already exists: `override` or `skip` |
| `--verbose` | off | Show the backend's native init logs (debugging) |

`single` only: `--out PATH`.
`batch` only: `--recursive` (scan nested subfolders at any depth, e.g.
`Show/Season 1/ep.mkv` or `Show/Recommendation/Season 1/ep.mkv`), `--force` (re-process
even if output exists). Files ending in `_thuyetminh`/`.ttstmp` are always skipped.

---

## 7. Troubleshooting

- **`ModuleNotFoundError: No module named 'srt'`** — run `python -m pip install srt pydub vieneu`.
- **VieNeu fails on init / phonemizer error** — eSpeak NG isn't installed (`brew install espeak`).
- **"Could not auto-detect a Vietnamese subtitle"** — the script lists the streams;
  pick one with `--sub-index N`.
- **"Subtitle is image-based (PGS/VOBSUB)"** — image subtitles need OCR and aren't
  supported; choose a text subtitle track with `--sub-index`.
- **Mixed track drifts / film audio runs fast / no sound near the end** — the chosen
  film audio is first decoded into a clean, exact-length WAV (`+genpts`,
  `aresample=async=1`, pad/trim to the video duration) and the mix is built from that, so
  it can't drift even when the source track is a corrupt Opus stream that drops packets.
  The original track is stream-copied untouched.
- **`Error parsing Opus packet header` warnings** — printed by ffmpeg while reading a
  flaky Opus source (or a leftover narration track from an older run). They are warnings,
  not failures: the original track is copied byte-for-byte (plays fine, as before) and the
  voiceover is built from the repaired WAV. Leftover narration tracks disappear after one
  idempotent re-run.
- **Wrong audio track chosen as the mix base** — the base is selected by: drop Vietnamese
  tracks (so a voiceover is never mixed onto Vietnamese audio) and tool-made narration,
  then prefer `--audio-lang` (default English) by `language` tag, then by the full language
  name as a whole word in the title, then the film's default track. Vietnamese is matched
  by tag (`vie`/`vi`) or title words (`viet`/`việt`/`lồng tiếng`/`thuyết minh`); ambiguous
  bare codes like `vi`/`tv` are not matched inside titles. Override with `--audio-index N`.
- **Re-dubbing files that already have a voiceover** — pass `--if-exists skip` to leave
  files that already contain a tool-made Vietnamese narration track untouched (useful when
  re-running `batch --in-place` over a season); the default `override` replaces it.
- **Narration sounds too slow** — VieNeu has no built-in rate control, so use
  `--speed 1.15` (pitch preserved via ffmpeg `atempo`). 1.1–1.25 is transparent for
  speech; above ~1.3 it starts to sound artificial. The cache stores natural-speed audio,
  so you can change `--speed` and re-run without re-synthesizing.
- **An all-caps line is spelled out letter by letter** — fixed: a cue whose letters are all
  uppercase (ignoring digits/symbols) is lowercased before synthesis so it's read as words.
- **`ValueError: No valid speech tokens found in the output`** — VieNeu couldn't vocalize
  a particular cue (often a line that is only punctuation/symbols, or an odd number-only
  string). Such cues are now skipped automatically (the slot stays silent) and a
  `skipped cue #N` warning is printed; one bad cue no longer aborts the file, and `batch`
  keeps going to the next file regardless of the error.
- **Subtitle formatting tags** — `<i>…</i>`, `<b>`, `<font …>`, ASS override blocks like
  `{\an8}`/`{\i1}`, ASS line breaks (`\N`) and HTML entities (`&amp;`, `&#39;`) are all
  stripped/decoded before synthesis, so e.g. `đây là <i>chữ nghiêng</i>` reads as
  `đây là chữ nghiêng`.
- **Numbers/symbols** — `%`, `°C`, `°F`, `km/h`, `kg`, `km`, `&`, `+`, `=`, `~` are
  expanded to Vietnamese words; `:` `/` `-` are kept so times/dates/ranges survive.
  Extend the `_EXPAND` map in `srt_to_voiceover.py` for anything else. Tricky large
  numbers still read best when written as words.

---

## 8. For an AI agent

Deterministic recipe (no interactive prompts):

```bash
# 1. Install (Python >= 3.9)
python -m pip install -r requirements.txt   # or: python -m pip install vieneu srt pydub
#    Ensure ffmpeg + ffprobe and espeak-ng are installed at the OS level.

# 2. Single file -> writes movie_thuyetminh.mkv
python mkv_voiceover.py single "<INPUT>.mkv" \
    --voice "Ngọc Lan" --emotion storytelling \
    --sub-lang vie --audio-lang eng \
    --merge-gap 34 --cache-dir .ttscache --workers 2

# 3. Whole folder, overwriting originals in place
python mkv_voiceover.py batch "<FOLDER>" --recursive \
    --voice "Ngọc Lan" --cache-dir .ttscache --in-place --workers 2

# 4. Inspect the result
ffprobe -v error -show_streams -select_streams a "<INPUT>_thuyetminh.mkv"
```

Conventions: the new audio track is tagged `language=vie`, titled "Thuyết minh (TTS)",
and is **not** default unless `--set-default` is passed.

Log format (stable, easy to parse): each step prints `[i/6] <name>` then `      done (Xs)`;
synthesis adds a live `done/total (…%) … eta …` line; the run ends with
`DONE <output-path> (Xs total)`. To know a file finished, wait for its `DONE ` line.

Exit codes: `single` exits non-zero on any fatal error (returns 0 on success). `batch`
continues past per-file failures and prints `Batch done: N ok, M skipped, K failed`
with a `Failed:` list. Add `--verbose` to surface the backend's native init logs when
debugging.
