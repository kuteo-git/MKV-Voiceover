#!/usr/bin/env python3
"""mkv_voiceover.py

End-to-end: take an .mkv that embeds a Vietnamese subtitle track, turn that
subtitle into a VieNeu-TTS voiceover, and add it back into the .mkv as an extra
audio track. The original video/audio/subtitles are kept untouched (stream copy),
so a player can simply switch the audio channel to the "Thuyết minh" track.

    python mkv_voiceover.py movie.mkv --voice "Trọng Hữu" --emotion storytelling \
        --merge-gap 350 --cache-dir .ttscache

Pipeline:  probe MKV -> pick Vietnamese sub -> extract to .srt
           -> synth narration (reuses srt_to_voiceover core)
           -> append narration as a tagged audio track (ffmpeg -c copy).

Performance notes
-----------------
  * The TTS step is the only heavy part; muxing is a near-instant stream copy.
  * VieNeu-TTS v3 Turbo runs torch-free via ONNX on CPU (PyTorch on GPU). This
    script optimizes by doing LESS / more-parallel work:
      - per-cue caching (--cache-dir) makes re-runs/retries free,
      - cue merging (--merge-gap) reduces the number of model calls,
      - parallel synthesis (--workers) spreads cues across processes.
  * Only the new narration track is encoded (Opus); everything else is copied.

Requires: ffmpeg + ffprobe on PATH, plus the part-1 deps (vieneu, srt, pydub)
and eSpeak NG. Keep this file next to srt_to_voiceover.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import List, Optional

from srt_to_voiceover import (  # core reused from part 1
    Synthesizer,
    merge_cues,
    parse_srt,
    prefill_cache_parallel,
    probe_duration_ms,
    render_narration,
)

# Subtitle codecs we can turn into text. Image-based subs (PGS/VOBSUB/DVB) would
# need OCR and are explicitly rejected.
_TEXT_SUB_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}
_IMAGE_SUB_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _stage(i: int, total: int, name: str):
    """Print a numbered START/DONE pair around a step, with timing.

    Output is stable and easy to parse: '[i/total] name' then '      done (Xs)'.
    """
    print(f"[{i}/{total}] {name}", flush=True)
    t = time.monotonic()
    try:
        yield
    except BaseException:
        print(f"      x failed", flush=True)
        raise
    else:
        print(f"      done ({time.monotonic() - t:.1f}s)", flush=True)


def _run_ffmpeg(cmd: list) -> None:
    """Run an ffmpeg command quietly; surface stderr only if it fails."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise subprocess.CalledProcessError(proc.returncode, cmd)


# --------------------------------------------------------------------------- #
# ffprobe helpers
# --------------------------------------------------------------------------- #
def _ffprobe_streams(mkv: Path, stream_type: str) -> List[dict]:
    """Return the ordered list of streams of a given type ('s', 'a', 'v')."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream_type,
         "-show_entries", "stream=index,codec_name,disposition:stream_tags=language,title",
         "-of", "json", str(mkv)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out).get("streams", [])


def _looks_vietnamese(stream: dict, want_lang: str) -> bool:
    tags = {k.lower(): str(v) for k, v in (stream.get("tags") or {}).items()}
    lang = tags.get("language", "").lower()
    title = unicodedata.normalize("NFD", tags.get("title", "").lower())
    title_ascii = "".join(c for c in title if not unicodedata.combining(c))
    return lang in {want_lang, "vi", "vie"} or "viet" in title_ascii


def find_vietnamese_sub(mkv: Path, want_lang: str, override: Optional[int]) -> tuple[int, str]:
    """Return (subtitle-relative index for -map 0:s:N, codec_name).

    `override` forces a specific subtitle-relative index, bypassing detection.
    """
    subs = _ffprobe_streams(mkv, "s")
    if not subs:
        sys.exit("No subtitle streams found in the MKV.")

    if override is not None:
        if not 0 <= override < len(subs):
            sys.exit(f"--sub-index {override} out of range (0..{len(subs)-1}).")
        chosen = override
    else:
        matches = [i for i, s in enumerate(subs) if _looks_vietnamese(s, want_lang)]
        if not matches:
            listing = "\n".join(
                f"  s:{i}  {s.get('codec_name')}  {s.get('tags', {})}"
                for i, s in enumerate(subs)
            )
            sys.exit("Could not auto-detect a Vietnamese subtitle. Pick one with "
                     f"--sub-index N:\n{listing}")
        chosen = matches[0]

    codec = subs[chosen].get("codec_name", "")
    if codec in _IMAGE_SUB_CODECS:
        sys.exit(f"Subtitle s:{chosen} is image-based ({codec}); OCR needed, "
                 "not supported. Choose a text subtitle with --sub-index.")
    if codec not in _TEXT_SUB_CODECS:
        print(f"Warning: unrecognized subtitle codec '{codec}', attempting extract anyway.")
    return chosen, codec


def count_audio_streams(mkv: Path) -> int:
    return len(_ffprobe_streams(mkv, "a"))


# Canonical language codes for matching (handles code/name variants robustly).
_CANON = {
    "en": "eng", "eng": "eng", "english": "eng",
    "ja": "jpn", "jp": "jpn", "jpn": "jpn", "japanese": "jpn",
    "vi": "vie", "vie": "vie", "vietnamese": "vie",
    "ko": "kor", "kor": "kor", "korean": "kor",
    "zh": "zho", "zho": "zho", "chi": "zho", "chinese": "zho",
    "fr": "fra", "fra": "fra", "fre": "fra", "french": "fra",
    "es": "spa", "spa": "spa", "spanish": "spa",
}
# Full names used for whole-word title matching (never short codes, which would
# wrongly match substrings like 'en' inside 'mrbenhien').
_FULLNAME = {"eng": "english", "jpn": "japanese", "vie": "vietnamese",
             "kor": "korean", "zho": "chinese", "fra": "french", "spa": "spanish"}


def _canon_lang(s: str) -> str:
    s = (s or "").strip().lower()
    return _CANON.get(s, s)


def _fold(s: str) -> str:
    """Lowercase + strip diacritics, so 'Việt'/'Lồng Tiếng' -> 'viet'/'long tieng'."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# Title markers that reliably indicate a Vietnamese track (folded, no diacritics).
# Bare codes 'vi'/'tv' are intentionally NOT matched in titles (too ambiguous as
# substrings); the language tag handles 'vi'.
_VI_TITLE_MARKERS = ("viet", "long tieng", "thuyet minh", "loi viet", "vietsub")


def _is_vietnamese(stream: dict) -> bool:
    """True if a stream is a Vietnamese audio track (dub, narration, etc.)."""
    tags = {k.lower(): str(v) for k, v in (stream.get("tags") or {}).items()}
    if _canon_lang(tags.get("language", "")) == "vie":
        return True
    title = _fold(tags.get("title", ""))
    return any(m in title for m in _VI_TITLE_MARKERS)


def _is_prior_narration(stream: dict) -> bool:
    """True if a stream looks like a narration track this tool added before.

    Matches our marker (Vietnamese + title contains 'tts'/'thuyet minh'), so a
    real Vietnamese dub with some other title is NOT treated as ours.
    """
    tags = {k.lower(): str(v) for k, v in (stream.get("tags") or {}).items()}
    lang = tags.get("language", "").lower()
    title = _fold(tags.get("title", ""))
    return lang in {"vie", "vi"} and ("tts" in title or "thuyet minh" in title)


def has_prior_narration(mkv: Path) -> bool:
    """Whether the MKV already has an audio track this tool added before."""
    return any(_is_prior_narration(s) for s in _ffprobe_streams(mkv, "a"))


def find_audio_stream(mkv: Path, want_lang: str, override: Optional[int]) -> int:
    """Return the audio-relative index (for 0:a:N) to mix the narration onto.

    Policy (a Vietnamese voiceover should never be mixed onto Vietnamese audio):
        0. --audio-index forces a specific track.
        1. Drop Vietnamese tracks and tracks this tool added before.
        2. Among the rest, prefer `want_lang` (default English) by language tag,
           then by the full language name as a whole word in the title.
        3. Else use the film's default audio track (disposition=default), or the
           first remaining track.
        4. If every track is Vietnamese, fall back to the first one (with a warning).
    """
    auds = _ffprobe_streams(mkv, "a")
    if not auds:
        sys.exit("No audio streams found in the MKV.")
    if override is not None:
        if not 0 <= override < len(auds):
            sys.exit(f"--audio-index {override} out of range (0..{len(auds)-1}).")
        return override

    def tag(i: int, key: str) -> str:
        return {k.lower(): str(v) for k, v in (auds[i].get("tags") or {}).items()}.get(key, "")

    def is_default(i: int) -> bool:
        return bool((auds[i].get("disposition") or {}).get("default"))

    want_c = _canon_lang(want_lang)
    usable = [i for i, s in enumerate(auds)
              if not _is_prior_narration(s) and not _is_vietnamese(s)]

    if not usable:  # nothing but Vietnamese / our own tracks
        nonnar = [i for i, s in enumerate(auds) if not _is_prior_narration(s)]
        fb = nonnar[0] if nonnar else 0
        print(f"Warning: only Vietnamese audio found; mixing onto a:{fb}.")
        return fb

    # 1) preferred language by tag
    for i in usable:
        if _canon_lang(tag(i, "language")) == want_c:
            return i
    # 2) preferred language by whole-word full name in title
    full = _FULLNAME.get(want_c)
    if full:
        for i in usable:
            if re.search(rf"\b{full}\b", tag(i, "title").lower()):
                return i
    # 3) the film's default audio track among the usable ones
    for i in usable:
        if is_default(i):
            print(f"Warning: no '{want_lang}' track; using the default audio a:{i}.")
            return i
    # 4) first usable track
    print(f"Warning: no '{want_lang}' track; using a:{usable[0]}.")
    return usable[0]


def measure_lufs(source: Path, map_arg: Optional[str], seconds: Optional[int]) -> Optional[float]:
    """Measure integrated loudness (LUFS, EBU R128) of an audio source.

    Uses ffmpeg's loudnorm analysis (print_format=json). `seconds` limits the
    analysis window for speed; `map_arg` (e.g. '0:a:1') selects one stream.
    Returns the integrated loudness, or None if measurement failed.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostats"]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-i", str(source)]
    if map_arg:
        cmd += ["-map", map_arg]
    cmd += ["-af", "loudnorm=print_format=json", "-vn", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    err = proc.stderr
    try:
        block = err[err.rindex("{"): err.rindex("}") + 1]
        return float(json.loads(block)["input_i"])
    except (ValueError, KeyError):
        return None


# --------------------------------------------------------------------------- #
# ffmpeg operations
# --------------------------------------------------------------------------- #
def extract_subtitle(mkv: Path, sub_rel_index: int, out_srt: Path) -> None:
    """Extract one subtitle stream to .srt (ffmpeg converts ass/vtt -> srt)."""
    _run_ffmpeg(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(mkv), "-map", f"0:s:{sub_rel_index}", str(out_srt)]
    )


def prepare_film_audio(mkv: Path, film_audio_index: int, film_seconds: float,
                       out_wav: Path) -> None:
    """Decode the chosen audio track to a clean, exactly-video-length 48k stereo WAV.

    This isolates a fragile/corrupt source (e.g. Opus tracks that drop packets and
    cause progressive drift) into one robust pass: +genpts regenerates timestamps,
    aresample async fills gaps with silence, and apad/atrim pin the result to the
    exact video duration. The mix then uses this clean WAV, so it cannot drift.
    The original track in the MKV is left untouched (it is stream-copied later).
    """
    dur = f"{film_seconds:.3f}"
    _run_ffmpeg(
        ["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts",
         "-i", str(mkv), "-map", f"0:a:{film_audio_index}", "-vn",
         "-af", (f"aresample=async=1:first_pts=0,"
                 f"aformat=sample_rates=48000:channel_layouts=stereo,"
                 f"apad=whole_dur={dur},atrim=end={dur}"),
         "-c:a", "pcm_s16le", str(out_wav)]
    )


def add_narration_track(
    mkv: Path,
    narration: Path,
    film_audio: Path,
    out_mkv: Path,
    *,
    film_seconds: float,
    audio_codec: str,
    bitrate: str,
    title: str,
    make_default: bool,
    mix: bool,
    duck_mode: str,
    duck_db: float,
    voice_db: float,
    target_lufs: Optional[float],
) -> None:
    """Append a new audio track, mixing the clean film WAV with the narration.

    Inputs: 0 = source MKV (for video/original-audio/subs/attachment copy),
            1 = narration WAV, 2 = clean film-audio WAV (from prepare_film_audio).
    Both audio inputs are already well-formed, so the mix can't drift. If
    `target_lufs` is set the final mix is loudness-normalized (EBU R128).

    mix + duck_mode='auto'   -> sidechain ducking (film dips only under the voice).
    mix + duck_mode='static' -> film constantly lowered by `duck_db`.
    mix=False                -> narration only.
    """
    # Map only original audio tracks; drop any narration this tool added before,
    # so re-runs (especially --in-place) stay idempotent instead of stacking tracks.
    audio_streams = _ffprobe_streams(mkv, "a")
    keep = [i for i, s in enumerate(audio_streams) if not _is_prior_narration(s)]
    new_idx = len(keep)  # appended track sits after the kept originals

    dur = f"{film_seconds:.3f}"
    src = "[2:a]aformat=sample_rates=48000:channel_layouts=stereo"
    voice = f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={voice_db}dB"
    norm = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11," if target_lufs is not None else ""
    fit = f"apad=whole_dur={dur},atrim=end={dur}"  # belt-and-suspenders length pin
    tail = f"[mx]{norm}alimiter=limit=0.95[na]"

    if not mix:
        narr_norm = f",loudnorm=I={target_lufs}:TP=-1.5:LRA=11" if target_lufs is not None else ""
        filtergraph = f"{voice}{narr_norm},{fit}[na]"
    elif duck_mode == "auto":
        filtergraph = (
            f"{src}[film];"
            f"{voice},asplit=2[voice][key];"
            f"[film][key]sidechaincompress="
            f"threshold=0.05:ratio=8:attack=20:release=350:detection=rms[duck];"
            f"[duck][voice]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[m0];"
            f"[m0]{fit}[mx];"
            f"{tail}"
        )
    else:  # static
        filtergraph = (
            f"{src},volume={duck_db}dB[film];"
            f"{voice}[voice];"
            f"[film][voice]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[m0];"
            f"[m0]{fit}[mx];"
            f"{tail}"
        )

    # 0:V (uppercase) = real video only, EXCLUDING attached pictures (cover art).
    # A cover-art mjpeg stream has a single packet that never advances, so with
    # -max_interleave_delta 0 ffmpeg would buffer the whole file in RAM waiting for
    # it — OOM/SIGKILL on long high-bitrate remuxes. Dropping it avoids that.
    maps: list[str] = ["-map", "0:V"]
    for i in keep:
        maps += ["-map", f"0:a:{i}"]
    maps += ["-map", "0:s?", "-map", "0:t?", "-map", "[na]"]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mkv), "-i", str(narration), "-i", str(film_audio),
        "-filter_complex", filtergraph,
        *maps,
        "-c", "copy",
        f"-c:a:{new_idx}", audio_codec, f"-b:a:{new_idx}", bitrate,
        f"-metadata:s:a:{new_idx}", "language=vie",
        f"-metadata:s:a:{new_idx}", f"title={title}",
        f"-disposition:a:{new_idx}", "default" if make_default else "0",
        "-max_interleave_delta", "0",   # avoid MKV playback stalling on sparse streams
        str(out_mkv),
    ]
    _run_ffmpeg(cmd)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _resolve_voice_db(args: argparse.Namespace, film_audio: Path, narration: Path) -> float:
    """Pick the narration gain: measured (auto-balance) or the fixed --voice-db.

    Auto-balance measures the integrated loudness of both the (clean) film audio
    and the narration, then sets the voice gain so the narrator sits
    `--target-offset` LU above the film. Falls back to --voice-db on failure.
    """
    if not args.auto_balance:
        return args.voice_db

    film_lufs = measure_lufs(film_audio, None, args.analyze_seconds)
    narr_lufs = measure_lufs(narration, None, None)
    if film_lufs is None or narr_lufs is None:
        print("      loudness measurement failed; using --voice-db.")
        return args.voice_db

    gain = (film_lufs + args.target_offset) - narr_lufs
    gain = max(-12.0, min(12.0, gain))  # clamp to a sane range
    print(f"      film {film_lufs:.1f} LUFS, narration {narr_lufs:.1f} LUFS "
          f"-> voice gain {gain:+.1f} dB (offset {args.target_offset:+.1f} LU)")
    return gain


def resolve_cache_dir(args: argparse.Namespace, mkv: Path) -> Optional[Path]:
    """Resolve --cache-dir. 'auto' places it next to the MKV; else use as given."""
    if not args.cache_dir:
        return None
    if args.cache_dir == "auto":
        return mkv.parent / ".ttscache"
    return Path(args.cache_dir)


def process_file(mkv: Path, args: argparse.Namespace, *,
                 out_override: Optional[Path]) -> tuple[Optional[Path], Optional[Path]]:
    """Run the full pipeline for one MKV.

    Returns (output_path, cache_dir). output_path is None if the file was skipped
    because it already has a narration track and --if-exists=skip.
    """
    if not mkv.exists():
        sys.exit(f"File not found: {mkv}")

    cache_dir = resolve_cache_dir(args, mkv)

    if args.if_exists == "skip" and has_prior_narration(mkv):
        print("Skipped: already has a Vietnamese narration track (--if-exists skip).")
        return None, cache_dir

    total = 6
    t0 = time.monotonic()

    with _stage(1, total, "Detect streams"):
        sub_idx, codec = find_vietnamese_sub(mkv, args.sub_lang, args.sub_index)
        film_idx = find_audio_stream(mkv, args.audio_lang, args.audio_index)
        film_ms = probe_duration_ms(mkv)
        print(f"      subtitle s:{sub_idx} ({codec}) | mix base a:{film_idx} | "
              f"length {film_ms/1000:.0f}s")

    if args.in_place:
        out_mkv, final = mkv.with_name(mkv.stem + ".ttstmp.mkv"), mkv
    else:
        final = out_override or mkv.with_name(mkv.stem + "_thuyetminh.mkv")
        out_mkv = final

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        with _stage(2, total, "Extract subtitle"):
            srt_path = tmp_dir / "vi.srt"
            extract_subtitle(mkv, sub_idx, srt_path)
            cues = parse_srt(srt_path)
            if not cues:
                sys.exit("Extracted subtitle had no usable cues.")
            raw_n = len(cues)
            cues = merge_cues(cues, args.merge_gap)
            print(f"      {raw_n} cues" + (f" -> {len(cues)} after merge" if len(cues) != raw_n else ""))

        synth = Synthesizer(
            emotion=args.emotion,
            voice_id=args.voice,
            ref_audio=Path(args.clone_audio) if args.clone_audio else None,
            ref_text=args.clone_text,
            quiet=not args.verbose,
        )

        with _stage(3, total, "Synthesize narration"):
            if cache_dir and args.workers > 1:
                prefill_cache_parallel(cues, synth, cache_dir=cache_dir, workers=args.workers)
            track = render_narration(
                cues, synth,
                max_speed=args.max_speed,
                floor_ms=film_ms,
                work_dir=tmp_dir,
                cache_dir=cache_dir,
                speed=args.speed,
            )
            narration_wav = tmp_dir / "narration.wav"
            track.export(narration_wav, format="wav")

        with _stage(4, total, "Repair film audio"):
            # clean, exact-length WAV so the mix can't drift even if the source drops packets
            film_wav = tmp_dir / "film.wav"
            prepare_film_audio(mkv, film_idx, film_ms / 1000.0, film_wav)

        with _stage(5, total, "Balance loudness"):
            if args.auto_balance:
                voice_db = _resolve_voice_db(args, film_wav, narration_wav)
            else:
                voice_db = args.voice_db
                print(f"      auto-balance off; voice gain {voice_db:+.1f} dB")

        with _stage(6, total, "Mux voiceover track"):
            add_narration_track(
                mkv, narration_wav, film_wav, out_mkv,
                film_seconds=film_ms / 1000.0,
                audio_codec=args.audio_codec,
                bitrate=args.bitrate,
                title=args.track_title,
                make_default=args.set_default,
                mix=not args.no_mix,
                duck_mode=args.duck_mode,
                duck_db=args.duck_db,
                voice_db=voice_db,
                target_lufs=(args.target_lufs if args.normalize else None),
            )
            if args.in_place:
                os.replace(out_mkv, final)

    print(f"DONE  {final}  ({time.monotonic() - t0:.1f}s total)")
    return final, cache_dir


def _clean_cache(cache_dir: Optional[Path]) -> None:
    if cache_dir and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Cleaned cache: {cache_dir}")


def run_single(args: argparse.Namespace) -> None:
    try:
        result, cache_dir = process_file(
            Path(args.mkv), args, out_override=Path(args.out) if args.out else None)
    except subprocess.CalledProcessError as e:
        sys.exit(f"FAILED: ffmpeg error ({e.returncode})")
    if args.clean_cache:
        _clean_cache(cache_dir)
    if result is not None:
        print("Tip: switch the audio track in your player to the Vietnamese voiceover.")


def run_batch(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    pattern = "**/*.mkv" if args.recursive else "*.mkv"
    files = sorted(f for f in folder.glob(pattern)
                   if not f.stem.endswith(("_thuyetminh", ".ttstmp")))
    if not files:
        sys.exit("No .mkv files found.")

    print(f"Found {len(files)} MKV file(s).\n")
    ok, skipped, failed = 0, 0, []
    cache_dirs: set[Path] = set()
    for i, mkv in enumerate(files, 1):
        out = mkv.with_name(mkv.stem + "_thuyetminh.mkv")
        try:
            rel = mkv.relative_to(folder)
        except ValueError:
            rel = mkv.name
        print(f"[file {i}/{len(files)}] {rel}")
        if not args.in_place and not args.force and out.exists():
            print("  skip (output exists; use --force to redo)")
            skipped += 1
            continue
        try:
            result, cache_dir = process_file(mkv, args, out_override=out)
            if cache_dir:
                cache_dirs.add(cache_dir)
            if result is None:
                skipped += 1
            else:
                ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: ffmpeg error ({e.returncode})")
            failed.append(mkv.name)
        except KeyboardInterrupt:
            raise
        except SystemExit as e:  # process_file uses sys.exit for fatal per-file issues
            print(f"  FAILED: {e}")
            failed.append(mkv.name)
        except Exception as e:  # any other per-file error: log and keep going
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(mkv.name)
        print()

    if args.clean_cache:  # clean once at the end so per-file retries still benefit
        for c in cache_dirs:
            _clean_cache(c)

    print(f"Batch done: {ok} ok, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))


def _add_common_args(p: argparse.ArgumentParser) -> None:
    # subtitle selection
    p.add_argument("--sub-lang", default="vie", help="Subtitle language tag to match (default vie)")
    p.add_argument("--sub-index", type=int, help="Force subtitle-relative index (0:s:N), skip detection")
    # audio (mix base) selection
    p.add_argument("--audio-lang", default="eng",
                   help="Preferred mix-base language among non-Vietnamese tracks "
                        "(default eng); Vietnamese tracks are always excluded as the base")
    p.add_argument("--audio-index", type=int, help="Force audio-relative index (0:a:N) for the mix base")
    # voice
    p.add_argument("--voice", help="Preset voice id, e.g. 'Ngọc Linh' or 'Trọng Hữu'")
    p.add_argument("--emotion", default="natural", help="'natural' or 'storytelling'")
    p.add_argument("--clone-audio", help="3-5s reference .wav to clone narrator voice")
    p.add_argument("--clone-text", help="Exact transcript of the reference clip")
    # timing / performance
    p.add_argument("--max-speed", type=float, default=1.5, help="Max speed-up to fit a slot")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Global pitch-preserving narration speed (1.1-1.25 transparent; VieNeu reads a bit slow)")
    p.add_argument("--merge-gap", type=int, default=0,
                   help="Merge cues with gaps <= this (ms) to cut TTS calls (default 0=off)")
    p.add_argument("--cache-dir",
                   help="Cache per-cue synthesis for fast re-runs. Use 'auto' to place "
                        "it next to each MKV (<folder>/.ttscache)")
    p.add_argument("--clean-cache", action="store_true",
                   help="Delete the cache dir after finishing (end of batch)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel TTS worker processes to pre-synthesize cues within "
                        "each file (needs --cache-dir; default 1). Speeds up a single "
                        "long file; each worker loads its own model (~1-2 GB RAM).")
    # loudness / balance
    p.add_argument("--auto-balance", action=argparse.BooleanOptionalAction, default=True,
                   help="Measure film+narration loudness and auto-set voice gain (default on)")
    p.add_argument("--target-offset", type=float, default=3.0,
                   help="LU the narrator sits above the film when auto-balancing (default +3)")
    p.add_argument("--analyze-seconds", type=int, default=300,
                   help="Seconds of film audio to analyze for loudness (default 300)")
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="Loudness-normalize the final mix to --target-lufs (default on)")
    p.add_argument("--target-lufs", type=float, default=-16.0,
                   help="Integrated loudness target for the final mix (default -16 LUFS)")
    p.add_argument("--voice-db", type=float, default=0.0,
                   help="Manual narration gain when --no-auto-balance (default 0)")
    p.add_argument("--duck-mode", choices=["auto", "static"], default="auto",
                   help="'auto' ducks film only while narrator speaks (default); 'static' lowers it constantly")
    p.add_argument("--duck-db", type=float, default=-9.0,
                   help="dB applied to film audio in 'static' duck mode (default -9)")
    # output track
    p.add_argument("--audio-codec", default="libopus", help="Codec for narration track (default libopus)")
    p.add_argument("--bitrate", default="128k", help="Narration track bitrate (default 128k)")
    p.add_argument("--track-title", default="Thuyết minh (TTS)", help="Title metadata for the new track")
    p.add_argument("--set-default", action="store_true", help="Make narration the default audio track")
    p.add_argument("--no-mix", action="store_true", help="New track = narration only (no film audio)")
    p.add_argument("--in-place", action="store_true", help="Overwrite source .mkv in place (atomic swap)")
    p.add_argument("--if-exists", choices=["override", "skip"], default="override",
                   help="If a Vietnamese narration track already exists: 'override' (default, "
                        "replace it) or 'skip' (leave the file untouched)")
    p.add_argument("--verbose", action="store_true",
                   help="Show the backend's native init logs (debugging)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a Vietnamese TTS voiceover track to MKV files")
    sub = parser.add_subparsers(dest="command", required=True)

    sp_single = sub.add_parser("single", help="Process one .mkv file")
    sp_single.add_argument("mkv", help="Input .mkv with an embedded Vietnamese subtitle")
    sp_single.add_argument("--out", help="Output .mkv path (auto if omitted)")
    _add_common_args(sp_single)

    sp_batch = sub.add_parser("batch", help="Process every .mkv in a folder")
    sp_batch.add_argument("folder", help="Folder containing .mkv files")
    sp_batch.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    sp_batch.add_argument("--force", action="store_true", help="Re-process even if output exists")
    _add_common_args(sp_batch)

    args = parser.parse_args()
    if args.workers > 1 and not args.cache_dir:
        sys.exit("--workers > 1 requires --cache-dir (workers share results via the cache).")
    if args.command == "batch":
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()