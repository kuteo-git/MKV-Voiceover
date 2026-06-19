#!/usr/bin/env python3
"""srt_to_voiceover.py

Generate a Vietnamese voiceover (thuyết minh) from an .srt subtitle file using
VieNeu-TTS, keeping it roughly in sync with the original subtitle timeline.

Outputs:
  1. A standalone narration audio file (.wav / .mp3).
  2. (optional) The narration muxed back into the source video.

Pipeline:  parse SRT  ->  synth per cue  ->  fit each clip to its time slot
           ->  assemble onto a silent track  ->  export / mux into video.

Dependencies
------------
  pip install vieneu srt pydub
  System: ffmpeg + ffprobe on PATH, and eSpeak NG (required by VieNeu).
      Ubuntu/Debian:  sudo apt install ffmpeg espeak-ng
      macOS:          brew install ffmpeg espeak

Notes / caveats
---------------
  * VieNeu-TTS is a neural model with NO native duration control, so timeline
    sync is enforced here by time-stretching each clip with ffmpeg `atempo`
    (pitch-preserving). Stretch is capped (--max-speed); beyond the cap the clip
    is allowed to overflow into the following gap instead of sounding sped-up.
  * VieNeu v3 Turbo output is 48 kHz, watermarked by default.
  * Model licensing: VieNeu-TTS v3 Turbo = Apache-2.0 (commercial OK).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import srt  # type: ignore
from pydub import AudioSegment  # type: ignore

VIENEU_SAMPLE_RATE = 48_000  # VieNeu-TTS v3 Turbo outputs 48 kHz


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cue:
    """A single subtitle line, normalized to milliseconds."""

    index: int
    start_ms: int
    end_ms: int
    text: str


# Punctuation the TTS reads as pauses/intonation — everything else is stripped.
# Symbols expanded to Vietnamese words BEFORE stripping (longest keys first so
# 'km/h' wins over 'km'). Edit/extend this map for your content.
_EXPAND = {
    "km/h": " ki lô mét giờ ", "m/s": " mét trên giây ",
    "°C": " độ xê ", "°F": " độ ép ", "℃": " độ xê ", "℉": " độ ép ", "°": " độ ",
    "%": " phần trăm ", "‰": " phần nghìn ",
    "kg": " ki lô gam ", "km": " ki lô mét ",
    "&": " và ", "+": " cộng ", "=": " bằng ", "~": " xấp xỉ ",
}
# Kept punctuation. ':' '/' '-' are kept so times (12:30), dates (3/6/2024) and
# ranges (2020-2024) survive into the phonemizer instead of being deleted.
_PUNCT_KEEP = set(".,!?:/-")
# Other separators mapped to the nearest kept mark so pauses aren't lost.
_PAUSE_MAP = {";": ",", "…": ".", "—": ",", "–": ","}


def normalize_text(text: str) -> str:
    """Clean a cue for TTS.

    Keeps letters (incl. Vietnamese), digits, basic punctuation and the
    date/time separators ':' '/' '-'. Expands common symbols/units (%, °C, km/h…)
    to words first, strips emoji/markup, and collapses whitespace and repeats.
    """
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)                        # &amp; -> &, &#39; -> ' etc.
    text = re.sub(r"<[^>]+>", " ", text)              # HTML/SRT tags: <i>, </i>, <font ...>
    text = re.sub(r"\{[^}]*\}", " ", text)            # ASS/SSA override blocks: {\an8}, {\i1}
    text = re.sub(r"\\[Nnh]", " ", text)              # ASS line breaks / hard spaces: \N \n \h
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Decide ALL-CAPS on the ORIGINAL letters, before expansion injects lowercase
    # words (e.g. '%' -> 'phần trăm'); VieNeu would otherwise spell all-caps out.
    src_letters = [c for c in text if c.isalpha()]
    was_all_caps = bool(src_letters) and all(c.isupper() for c in src_letters)
    for key in sorted(_EXPAND, key=len, reverse=True):
        text = text.replace(key, _EXPAND[key])
    text = "".join(_PAUSE_MAP.get(ch, ch) for ch in text)
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in _PUNCT_KEEP)
    text = re.sub(r"\s+", " ", text)                 # collapse whitespace incl. newlines
    text = re.sub(r"([.,!?:/\-])\1+", r"\1", text)   # collapse repeated separators
    text = re.sub(r"\s+([.,!?:])", r"\1", text)       # no space before sentence punctuation
    text = text.strip()
    if was_all_caps:
        text = text.lower()
    return text


def parse_srt(path: Path) -> List[Cue]:
    """Parse an .srt file into a clean, ordered list of cues.

    Empty cues are dropped; cues are sorted by start time so downstream
    slot math (gap-to-next) stays correct even on malformed inputs.
    """
    raw = path.read_text(encoding="utf-8-sig")  # tolerate BOM from many editors
    cues: List[Cue] = []
    for sub in srt.parse(raw):
        text = normalize_text(sub.content)
        if not text:
            continue
        cues.append(
            Cue(
                index=sub.index or len(cues) + 1,
                start_ms=int(sub.start.total_seconds() * 1000),
                end_ms=int(sub.end.total_seconds() * 1000),
                text=text,
            )
        )
    cues.sort(key=lambda c: c.start_ms)
    return cues


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _suppress_native_stderr(enabled: bool = True):
    """Temporarily silence C-level stderr (fd 2), e.g. ONNX Runtime / Metal init spam.

    Python-level exceptions still propagate; only the noisy native logging is hidden.
    """
    if not enabled:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class Synthesizer:
    """Thin wrapper over the VieNeu SDK.

    Lazily initialized so that --help and SRT validation don't pay the model
    load cost. Voice selection precedence:
        1. preset voice id  (e.g. 'Ngọc Linh', 'Trọng Hữu') via get_preset_voice()
        2. cloned voice from a 3-5s reference clip
        3. model default
    `emotion` ('natural' | 'storytelling') is applied per inference (v3 Turbo).
    `quiet` hides the backend's native init logging.
    """

    def __init__(
        self,
        emotion: str = "natural",
        voice_id: Optional[str] = None,
        ref_audio: Optional[Path] = None,
        ref_text: Optional[str] = None,
        quiet: bool = True,
    ) -> None:
        if voice_id and ref_audio:
            raise ValueError("Use either --voice (preset) OR --clone-audio, not both.")
        if (ref_audio is None) != (ref_text is None):
            raise ValueError("Voice cloning needs BOTH --clone-audio and --clone-text.")
        self._emotion = emotion
        self._voice_id = voice_id
        self._ref_audio = str(ref_audio) if ref_audio else None
        self._ref_text = ref_text
        self._quiet = quiet
        self._tts = None  # type: ignore
        self._voice_data = None  # resolved preset payload, cached after first load

    def _engine(self):
        if self._tts is None:
            from vieneu import Vieneu  # imported here to keep startup light

            with _suppress_native_stderr(self._quiet):
                # v3 Turbo is the default mode; emotion is now passed per-inference
                # (see synth()), not at engine construction time.
                self._tts = Vieneu(mode="v3turbo")
                if self._voice_id:
                    self._voice_data = self._tts.get_preset_voice(self._voice_id)
        return self._tts

    def load(self) -> None:
        """Eagerly load the model (used to warm up before the progress loop)."""
        self._engine()

    def cache_key(self, text: str) -> str:
        """Stable hash of text + voice settings, used for per-cue caching."""
        sig = f"{self._emotion}|{self._voice_id}|{self._ref_audio}|{self._ref_text}|{text}"
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()

    def config(self) -> dict:
        """Picklable kwargs to reconstruct this Synthesizer in a worker process.

        The model itself is not picklable, so parallel workers rebuild a fresh
        Synthesizer from this config. The fields exactly mirror those folded into
        cache_key(), so a worker writes cache files the parent will recognize.
        """
        return {
            "emotion": self._emotion,
            "voice_id": self._voice_id,
            "ref_audio": self._ref_audio,
            "ref_text": self._ref_text,
            "quiet": self._quiet,
        }

    def synth(self, text: str, out_path: Path) -> AudioSegment:
        """Synthesize `text` to a 48 kHz wav and return it as an AudioSegment."""
        tts = self._engine()
        with _suppress_native_stderr(self._quiet):
            if self._voice_data is not None:
                audio = tts.infer(text=text, voice=self._voice_data, emotion=self._emotion)
            elif self._ref_audio:
                audio = tts.infer(text=text, ref_audio=self._ref_audio,
                                  ref_text=self._ref_text, emotion=self._emotion)
            else:
                audio = tts.infer(text=text, emotion=self._emotion)
            tts.save(audio, str(out_path))
        return AudioSegment.from_wav(out_path)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _atempo_chain(factor: float) -> List[str]:
    """Decompose a tempo factor into a chain of ffmpeg atempo filters.

    ffmpeg's atempo is only stable in [0.5, 2.0], so factors outside that
    range are split into a product of in-range steps.
    """
    if abs(factor - 1.0) < 1e-3:
        return []
    steps: List[float] = []
    f = factor
    while f > 2.0:
        steps.append(2.0)
        f /= 2.0
    while f < 0.5:
        steps.append(0.5)
        f /= 0.5
    steps.append(f)
    return [f"atempo={s:.6f}" for s in steps]


def _time_stretch(clip: AudioSegment, factor: float) -> AudioSegment:
    """Change tempo by `factor` while preserving pitch (ffmpeg atempo / WSOLA).

    factor > 1 speeds up, < 1 slows down. ~1.1-1.25 is transparent for speech.
    """
    if abs(factor - 1.0) < 1e-3:
        return clip
    chain = _atempo_chain(factor)
    if not chain:
        return clip
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        clip.export(src, format="wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(src), "-filter:a", ",".join(chain), str(dst)],
            check=True,
        )
        return AudioSegment.from_wav(dst)


def _fit_factor(dur_ms: float, slot_ms: int, max_speed: float) -> float:
    """Tempo factor needed to fit `dur_ms` into `slot_ms`, capped at max_speed.

    Returns 1.0 (no change) when the clip already fits or the slot is unknown.
    """
    if slot_ms <= 0 or dur_ms <= slot_ms:
        return 1.0
    return min(dur_ms / slot_ms, max_speed)


def fit_to_slot(clip: AudioSegment, slot_ms: int, max_speed: float) -> AudioSegment:
    """Speed up `clip` (pitch-preserving) so it fits `slot_ms`, capped at max_speed.

    If the required factor exceeds the cap, the clip is sped to the cap only and
    the remainder is allowed to overflow into the following gap.
    """
    factor = _fit_factor(len(clip), slot_ms, max_speed)
    if abs(factor - 1.0) < 1e-3:
        return clip
    return _time_stretch(clip, factor)


def assemble_track(
    placements: List[tuple[int, AudioSegment]],
    floor_ms: int = 0,
) -> AudioSegment:
    """Place each (start_ms, clip) onto a silent base of sufficient length.

    Sums all clips into one preallocated buffer instead of chaining pydub's
    overlay(): overlay() copies the entire (multi-hundred-MB) base each call, so
    chaining N clips is O(N * base_len). Here we write each clip once into a
    shared int32 accumulator, then saturate to int16 — matching overlay's
    clipping behaviour on the rare overlap, at a fraction of the cost.
    """
    total = max([floor_ms] + [start + len(clip) for start, clip in placements]) + 500
    base = AudioSegment.silent(duration=total, frame_rate=VIENEU_SAMPLE_RATE)
    if not placements:
        return base

    import numpy as np

    rate, channels, width = base.frame_rate, base.channels, base.sample_width
    acc = np.zeros(len(base.get_array_of_samples()), dtype=np.int32)
    for start_ms, clip in placements:
        # Match the base's format so samples line up frame-for-frame.
        clip = clip.set_frame_rate(rate).set_channels(channels).set_sample_width(width)
        samples = np.frombuffer(clip.raw_data, dtype=np.int16).astype(np.int32)
        pos = int(start_ms * rate / 1000) * channels
        end = min(pos + samples.shape[0], acc.shape[0])
        acc[pos:end] += samples[: end - pos]

    limit = 1 << (8 * width - 1)
    np.clip(acc, -limit, limit - 1, out=acc)
    return base._spawn(acc.astype(np.int16).tobytes())


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def probe_duration_ms(video: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(float(out) * 1000)


def mux_video(
    video: Path,
    narration: Path,
    out_path: Path,
    keep_original: bool,
    duck_db: float,
) -> None:
    """Combine narration with the source video.

    keep_original=True  -> mix narration over original audio (original ducked
                           by `duck_db`), classic thuyết minh feel.
    keep_original=False -> replace the audio track entirely with narration.
    """
    if keep_original:
        flt = (
            f"[0:a]volume={duck_db}dB[bg];"
            f"[bg][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video), "-i", str(narration),
            "-filter_complex", flt,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video), "-i", str(narration),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-shortest", str(out_path),
        ]
    subprocess.run(cmd, check=True)


def merge_cues(cues: List[Cue], gap_ms: int) -> List[Cue]:
    """Merge consecutive cues separated by <= gap_ms into one.

    Fewer TTS calls => faster, and joined sentences read more naturally.
    No-op when gap_ms <= 0.
    """
    if gap_ms <= 0 or not cues:
        return cues
    merged: List[Cue] = [cues[0]]
    for cue in cues[1:]:
        prev = merged[-1]
        if cue.start_ms - prev.end_ms <= gap_ms:
            joiner = "" if prev.text.endswith((".", ",", "!", "?")) else "."
            merged[-1] = Cue(
                index=prev.index,
                start_ms=prev.start_ms,
                end_ms=cue.end_ms,
                text=f"{prev.text}{joiner} {cue.text}".strip(),
            )
        else:
            merged.append(cue)
    return merged


def _fmt_secs(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def render_narration(
    cues: List[Cue],
    synth: "Synthesizer",
    *,
    max_speed: float,
    floor_ms: int,
    work_dir: Path,
    cache_dir: Optional[Path] = None,
    speed: float = 1.0,
    show_progress: bool = True,
) -> AudioSegment:
    """Synthesize every cue, fit it to its slot and assemble the timeline.

    If `cache_dir` is given, each cue's raw synthesis is memoized by a hash of
    its text + voice settings, so re-runs and retries skip already-rendered cues.
    `speed` applies a global pitch-preserving tempo change to every clip (the cache
    stores natural-speed audio, so changing speed doesn't require re-synthesis).
    A single-line progress indicator with elapsed time / ETA is printed unless
    `show_progress` is False.
    """
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve each cue's cache path once (the key is a sha1, recomputing it per
    # cue twice would be wasteful), then warm up the model only if something is
    # actually missing from the cache.
    cached_paths: List[Optional[Path]] = [
        cache_dir / f"{synth.cache_key(c.text)}.wav" if cache_dir else None for c in cues
    ]
    needs_synth = any(not (cp and cp.exists()) for cp in cached_paths)
    if needs_synth:
        if show_progress:
            print("  loading VieNeu model (first run downloads it)…", flush=True)
        synth.load()

    placements: List[tuple[int, AudioSegment]] = []
    total = len(cues)
    start_t = time.monotonic()
    n_synth = n_cache = n_skip = 0

    for i, cue in enumerate(cues):
        next_start = cues[i + 1].start_ms if i + 1 < total else cue.end_ms
        slot_ms = max(next_start - cue.start_ms, cue.end_ms - cue.start_ms)

        # Skip cues with nothing speakable (only punctuation/symbols); the TTS
        # model raises "No valid speech tokens" on these. The slot stays silent.
        if not any(ch.isalnum() for ch in cue.text):
            n_skip += 1
            continue

        raw_path = work_dir / f"cue_{cue.index}.wav"
        cached = cached_paths[i]
        if cached and cached.exists():
            clip = AudioSegment.from_wav(cached)
            n_cache += 1
        else:
            try:
                clip = synth.synth(cue.text, raw_path)
            except Exception as e:  # one bad cue must not abort the whole file/batch
                n_skip += 1
                sys.stdout.write(f"\n  warn: skipped cue #{cue.index}: {e}\n")
                sys.stdout.flush()
                continue
            if cached:
                clip.export(cached, format="wav")
            n_synth += 1

        # Combine the global `speed` and the slot-fit into a single tempo factor so
        # each clip is stretched at most once (each stretch is a separate ffmpeg
        # call). The cap still applies only to the fit portion, as before: global
        # speed first, then fit the resulting duration into the slot.
        factor = speed * _fit_factor(len(clip) / speed, slot_ms, max_speed)
        if abs(factor - 1.0) >= 1e-3:
            clip = _time_stretch(clip, factor)
        placements.append((cue.start_ms, clip))

        if show_progress:
            done = i + 1
            elapsed = time.monotonic() - start_t
            eta = (elapsed / done) * (total - done)
            skip_note = f" / skip {n_skip}" if n_skip else ""
            sys.stdout.write(
                f"\r  {done}/{total} ({100 * done / total:4.1f}%)  "
                f"synth {n_synth} / cache {n_cache}{skip_note}  "
                f"elapsed {_fmt_secs(elapsed)}  eta {_fmt_secs(eta)}   "
            )
            sys.stdout.flush()

    if show_progress:
        sys.stdout.write("\n")
        if n_skip:
            sys.stdout.write(f"  ({n_skip} cue(s) skipped)\n")
        sys.stdout.flush()

    return assemble_track(placements, floor_ms=floor_ms)


# --------------------------------------------------------------------------- #
# Parallel cache pre-fill
# --------------------------------------------------------------------------- #
def _progress_bar(done: int, total: int, start_t: float, *, label: str = "", width: int = 26) -> None:
    """Render an in-place '[####----] d/t (p%) elapsed/eta' slider to stdout.

    ETA is a linear extrapolation from the average time per completed item, so it
    settles once a few items are done. Caller is responsible for the trailing
    newline when finished.
    """
    frac = (done / total) if total else 1.0
    filled = int(round(width * frac))
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.monotonic() - start_t
    eta = (elapsed / done) * (total - done) if done else 0.0
    head = f"{label} " if label else ""
    sys.stdout.write(
        f"\r  {head}[{bar}] {done}/{total} ({100 * frac:4.1f}%)  "
        f"elapsed {_fmt_secs(elapsed)}  eta {_fmt_secs(eta)}   "
    )
    sys.stdout.flush()


# Per-worker model, loaded once by the pool initializer and reused across every
# task that worker handles (re-loading per cue would dwarf the synthesis cost).
_WORKER_SYNTH: Optional["Synthesizer"] = None


def _worker_init(config: dict, omp_threads: int) -> None:
    """ProcessPool initializer: cap CPU threads, then load this worker's model once."""
    if omp_threads:
        # Cap each worker to its share of cores. Must be set BEFORE the TTS backend
        # (onnxruntime / torch) initializes, else N workers each grab every core and
        # thrash. Safe here because vieneu is imported lazily inside Synthesizer.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(omp_threads)
    global _WORKER_SYNTH
    _WORKER_SYNTH = Synthesizer(**config)
    _WORKER_SYNTH.load()  # eager, so the model-load cost lands here, not in task #1


def _synth_one(task: tuple) -> bool:
    """Synthesize one (cache_key, text) into cache_dir/<key>.wav. Returns success.

    A failure on one cue is swallowed (so a single bad cue can't break the pool) —
    render_narration retries any gap left behind.
    """
    key, text, cache_dir_str = task
    out = Path(cache_dir_str) / f"{key}.wav"
    if out.exists():
        return True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            clip = _WORKER_SYNTH.synth(text, Path(tmp) / "w.wav")  # type: ignore[union-attr]
            clip.export(out, format="wav")
        return True
    except Exception:
        return False


def prefill_cache_parallel(
    cues: List[Cue],
    synth: "Synthesizer",
    *,
    cache_dir: Path,
    workers: int,
    show_progress: bool = True,
) -> int:
    """Synthesize missing cues into `cache_dir` using `workers` processes.

    Cue synthesis is embarrassingly parallel (each cue is an independent TTS
    call), so this pre-fills the per-cue cache across several processes — each
    with its own model — before the sequential render_narration pass, which then
    finds everything cached and only does the cheap fit/assemble work. This is
    what speeds up a SINGLE long file, where file-level parallelism can't help.

    Each cue is submitted as its own task and drained via as_completed, so a live
    progress bar with ETA can advance per cue (the model is loaded once per worker
    in the initializer, not per task). Returns the number synthesized OK. No-op
    (returns 0) when workers <= 1 or nothing is missing; render_narration then
    synthesizes serially as before.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Collect unique, speakable, not-yet-cached cues. Dedupe by cache key so the
    # same line (common after --merge-gap) isn't synthesized twice.
    seen: set[str] = set()
    items: List[tuple[str, str]] = []
    for c in cues:
        if not any(ch.isalnum() for ch in c.text):
            continue
        key = synth.cache_key(c.text)
        if key in seen or (cache_dir / f"{key}.wav").exists():
            continue
        seen.add(key)
        items.append((key, c.text))

    workers = max(1, min(workers, len(items)))
    if workers <= 1:
        return 0

    omp = max(1, (os.cpu_count() or workers) // workers)
    config = synth.config()
    tasks = [(key, text, str(cache_dir)) for key, text in items]
    total = len(tasks)

    if show_progress:
        print(f"  prefill: {total} cue(s) across {workers} worker(s) "
              f"({omp} thread(s) each); loading models…", flush=True)
    start = time.monotonic()
    done = ok = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(config, omp)
    ) as ex:
        futures = [ex.submit(_synth_one, t) for t in tasks]
        for fut in as_completed(futures):
            ok += bool(fut.result())
            done += 1
            if show_progress:
                _progress_bar(done, total, start, label="prefill")
    if show_progress:
        sys.stdout.write("\n")
        miss = total - ok
        note = f" ({miss} deferred to serial pass)" if miss else ""
        print(f"  prefill done: {ok}/{total} in {_fmt_secs(time.monotonic() - start)}{note}",
              flush=True)
    return ok


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_voiceover(args: argparse.Namespace) -> None:
    srt_path = Path(args.srt)
    cues = parse_srt(srt_path)
    if not cues:
        sys.exit("No usable subtitle cues found.")
    cues = merge_cues(cues, getattr(args, "merge_gap", 0))
    print(f"Parsed {len(cues)} cues from {srt_path.name}")

    synth = Synthesizer(
        emotion=args.emotion,
        voice_id=args.voice,
        ref_audio=Path(args.clone_audio) if args.clone_audio else None,
        ref_text=args.clone_text,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if getattr(args, "workers", 1) > 1 and not cache_dir:
        sys.exit("--workers > 1 requires --cache-dir (workers share results via the cache).")

    with tempfile.TemporaryDirectory() as tmp:
        floor = probe_duration_ms(Path(args.video)) if args.video else 0
        if cache_dir and getattr(args, "workers", 1) > 1:
            prefill_cache_parallel(cues, synth, cache_dir=cache_dir, workers=args.workers)
        track = render_narration(
            cues, synth,
            max_speed=args.max_speed,
            floor_ms=floor,
            work_dir=Path(tmp),
            cache_dir=cache_dir,
            speed=args.speed,
        )

        audio_out = Path(args.audio_out)
        fmt = audio_out.suffix.lstrip(".").lower() or "wav"
        track.export(audio_out, format=fmt)
        print(f"\nNarration audio -> {audio_out}")

        if args.video:
            video_out = Path(args.video_out or Path(args.video).with_name(
                Path(args.video).stem + "_thuyetminh" + Path(args.video).suffix))
            mux_video(
                Path(args.video), audio_out, video_out,
                keep_original=not args.replace_audio,
                duck_db=args.duck_db,
            )
            print(f"Video           -> {video_out}")


def main() -> None:
    p = argparse.ArgumentParser(description="SRT -> Vietnamese voiceover via VieNeu-TTS")
    p.add_argument("srt", help="Input .srt subtitle file")
    p.add_argument("--audio-out", default="narration.wav",
                   help="Narration output path (.wav or .mp3). Default: narration.wav")
    p.add_argument("--video", help="Optional source video to mux narration into")
    p.add_argument("--video-out", help="Muxed video output path (auto if omitted)")
    p.add_argument("--replace-audio", action="store_true",
                   help="Replace original audio entirely instead of mixing over it")
    p.add_argument("--duck-db", type=float, default=-12.0,
                   help="dB applied to original audio when mixing (default -12)")
    p.add_argument("--max-speed", type=float, default=1.5,
                   help="Max pitch-preserving speed-up to fit a slot (default 1.5)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Global pitch-preserving narration speed (1.1-1.25 transparent)")
    p.add_argument("--voice", help="Preset voice id, e.g. 'Ngọc Linh' or 'Trọng Hữu'")
    p.add_argument("--emotion", default="natural",
                   help="Voice emotion: 'natural' or 'storytelling' (default natural)")
    p.add_argument("--merge-gap", type=int, default=0,
                   help="Merge cues with gaps <= this (ms) to cut TTS calls (default 0=off)")
    p.add_argument("--cache-dir",
                   help="Directory to cache per-cue synthesis for fast re-runs")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel TTS worker processes to pre-synthesize cues "
                        "(needs --cache-dir; default 1). Speeds up a single long file.")
    p.add_argument("--clone-audio", help="3-5s reference .wav to clone narrator voice")
    p.add_argument("--clone-text", help="Exact transcript of the reference clip")
    build_voiceover(p.parse_args())


if __name__ == "__main__":
    main()