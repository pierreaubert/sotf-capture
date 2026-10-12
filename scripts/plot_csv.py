#!/usr/bin/env python3
"""Plot SPL overlays (log freq axis) with timing traces underneath.

Usage:
    python scripts/plot_csv.py a.csv b.csv -o out.png

Top panel: SPL vs frequency, x from 20 Hz to 20 kHz, y window 50 dB wide
(top edge auto-fit to the data, rounded up to 10 dB).
Bottom panel: timing view ("impulse response") from the sibling .wav of
each CSV (same directory and stem), or from .wav files passed directly.
"""

import argparse
import csv
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

MAX_POINTS = 50_000


@dataclass
class Trace:
    take: int
    label: str
    x: np.ndarray
    y: np.ndarray


def read_xy(path: Path, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file")
        if xcol not in reader.fieldnames or ycol not in reader.fieldnames:
            raise ValueError(
                f"{path}: columns {xcol!r}/{ycol!r} not found "
                f"(has: {', '.join(reader.fieldnames)})"
            )
        xs: list[float] = []
        ys: list[float] = []
        for row in reader:
            try:
                x, y = float(row[xcol]), float(row[ycol])
            except (ValueError, TypeError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    if not xs:
        raise ValueError(f"{path}: no numeric data in {xcol!r}/{ycol!r}")
    return np.array(xs), np.array(ys)


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    """Read first channel of PCM16/24/32 or float32 WAV (incl. extensible)."""
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    fmt: bytes | None = None
    raw: bytes | None = None
    off = 12
    while off + 8 <= len(data):
        cid, csz = data[off : off + 4], struct.unpack("<I", data[off + 4 : off + 8])[0]
        body = data[off + 8 : off + 8 + csz]
        if cid == b"fmt ":
            fmt = bytes(body)
        elif cid == b"data" and raw is None:
            raw = bytes(body)
        off += 8 + csz + (csz % 2)
    if fmt is None or raw is None:
        raise ValueError(f"{path}: WAV missing fmt or data chunk")
    audio = struct.unpack("<H", fmt[0:2])[0]
    nch, srate = struct.unpack("<H", fmt[2:4])[0], struct.unpack("<I", fmt[4:8])[0]
    bps = struct.unpack("<H", fmt[14:16])[0]
    if audio == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: real format is in the sub-guid
        if len(fmt) < 40:
            raise ValueError(f"{path}: truncated extensible fmt chunk")
        audio = struct.unpack("<H", fmt[24:26])[0]
    nframes = len(raw) * 8 // max(bps * nch, 1)
    if audio == 1 and bps == 16:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif audio == 1 and bps == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 0x800000, v - 0x1000000, v)
        a = v.astype(np.float64) / float(0x800000)
    elif audio == 1 and bps == 32:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(0x80000000)
    elif audio == 3 and bps == 32:
        a = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    else:
        raise ValueError(f"{path}: unsupported WAV format={audio} bits={bps}")
    a = a[: nframes * nch].reshape(nframes, nch)[:, 0]
    return srate, np.ascontiguousarray(a)


def spl_ylim(peak: float, span: float = 50.0, step: float = 10.0) -> tuple[float, float]:
    top = math.ceil(peak / step) * step
    return top - span, top


def decimate(x: np.ndarray, y: np.ndarray, n: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, y
    stride = math.ceil(len(x) / n)
    return x[::stride], y[::stride]


def make_figure(
    top: list[Trace],
    bottom: list[Trace],
    xmin: float = 20.0,
    xmax: float = 20000.0,
    span: float = 50.0,
    spl_top: float | None = None,
    xlabel: str = "frequency_hz",
    ylabel: str = "spl_db",
    title: str | None = None,
) -> Figure:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))
    for t in top:
        x, y = decimate(t.x, t.y)
        ax_top.plot(x, y, label=t.label, color=colors[t.take % len(colors)])
    peak = max(float(np.max(t.y)) for t in top)
    hi = spl_top if spl_top is not None else spl_ylim(peak, span)[1]
    ax_top.set_xscale("log")
    ax_top.set_xlim(xmin, xmax)
    ax_top.set_ylim(hi - span, hi)
    ax_top.set_xlabel(xlabel)
    ax_top.set_ylabel(ylabel)
    if title:
        ax_top.set_title(title)
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend()
    for t in bottom:
        x, y = decimate(t.x, t.y)
        ax_bot.plot(x, y, label=t.label, color=colors[t.take % len(colors)])
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("amplitude")
    ax_bot.set_title("timing (impulse response)")
    ax_bot.grid(True, alpha=0.3)
    if bottom:
        ax_bot.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot SPL overlays with timing traces below.")
    ap.add_argument("files", nargs="+", help="CSV files (and/or WAV files for timing panel)")
    ap.add_argument("-o", "--output", default="plot.png", help="output image (default: plot.png)")
    ap.add_argument("--xcol", default="frequency_hz", help="X column (default: frequency_hz)")
    ap.add_argument("--ycol", default="spl_db", help="Y column (default: spl_db)")
    ap.add_argument("--xlabel", default=None, help="X label (default: x column name)")
    ap.add_argument("--ylabel", default=None, help="Y label (default: y column name)")
    ap.add_argument("--title", default=None, help="top plot title")
    ap.add_argument("--xmin", type=float, default=20.0, help="top x min (default: 20)")
    ap.add_argument("--xmax", type=float, default=20000.0, help="top x max (default: 20000)")
    ap.add_argument("--spl-span", type=float, default=50.0, help="SPL y width in dB (default: 50)")
    ap.add_argument("--spl-top", type=float, default=None, help="SPL y top in dB (default: auto)")
    ap.add_argument("--no-wav", action="store_true", help="skip the timing panel")
    args = ap.parse_args()

    top: list[Trace] = []
    bottom: list[Trace] = []
    take = 0
    for name in args.files:
        path = Path(name)
        if not path.is_file():
            print(f"error: not found: {name}", file=sys.stderr)
            sys.exit(1)
        if path.suffix.lower() == ".wav":
            if args.no_wav:
                continue
            try:
                sr, a = read_wav_mono(path)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                sys.exit(1)
            t = np.arange(len(a)) / sr
            bottom.append(Trace(take, path.stem, t, a))
            take += 1
            continue
        try:
            xs, ys = read_xy(path, args.xcol, args.ycol)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        top.append(Trace(take, path.stem, xs, ys))
        if not args.no_wav:
            wav = path.with_suffix(".wav")
            if wav.is_file():
                try:
                    sr, a = read_wav_mono(wav)
                except ValueError as e:
                    print(f"warning: {e}", file=sys.stderr)
                else:
                    t = np.arange(len(a)) / sr
                    bottom.append(Trace(take, path.stem, t, a))
            else:
                print(f"warning: no sibling WAV for {name}, timing skipped", file=sys.stderr)
        take += 1

    if not top:
        print("error: no CSV input", file=sys.stderr)
        sys.exit(1)
    fig = make_figure(
        top,
        [] if args.no_wav else bottom,
        xmin=args.xmin,
        xmax=args.xmax,
        span=args.spl_span,
        spl_top=args.spl_top,
        xlabel=args.xlabel or args.xcol,
        ylabel=args.ylabel or args.ycol,
        title=args.title,
    )
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
