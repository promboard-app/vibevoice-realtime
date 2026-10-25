"""README 그림 · 소리 — 노드가 만든 목소리 WAV 를 파형 · 스펙트로그램 그림(PNG)과 mp3 로.

사용:
    python -I -B docs/scripts/make_visuals.py <spec.json> <docs 폴더> <ko|en>
spec.json:
    {"title": {"ko": "...", "en": "..."},
     "items": [{"wav": "a.wav", "name": "clone-ko", "label": {"ko": "...", "en": "..."},
                "text": "읽힌 글", "meta": {"ko": "...", "en": "..."}}],
     "footer": {"ko": "...", "en": "..."}, "out": "hero.png"}
<docs>/<언어>/<out>(기본 hero.png) 와 <docs>/<언어>/audio/<name>.mp3 를 쓴다(mp3 는 두 언어가 같은 소리라도 언어 폴더마다 둔다).
필요한 것: numpy · scipy · matplotlib · Pillow · ffmpeg(PATH) · Pretendard 글꼴(Windows 사용자 글꼴 폴더 또는 README_FONT_DIR).
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageFont  # noqa: E402
from scipy.signal import stft  # noqa: E402

FONT_DIR = os.environ.get("README_FONT_DIR") or os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
BG, LINE, MUTED, PURPLE = "#171717", "#3a3a3a", "#a5a5a5", "#bb86fc"
W_IN, DPI = 12.0, 100
TEXT_PX = 13 * DPI / 72

for w in ("Regular", "SemiBold"):
    font_manager.fontManager.addfont(os.path.join(FONT_DIR, f"Pretendard-{w}.otf"))
plt.rcParams.update({
    "font.family": "Pretendard", "font.size": 12, "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": "white", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": LINE,
})
PIL_FONT = ImageFont.truetype(os.path.join(FONT_DIR, "Pretendard-Regular.otf"), round(TEXT_PX))


def read_wav(path):
    with wave.open(path) as w:
        fs, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: 16비트 PCM 만 읽는다")
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), fs


def wrap(text, max_px, max_lines=3):
    """글자 폭(Pretendard)으로 줄을 나눈다. 넘치면 마지막 줄 끝을 … 로."""
    lines, cur = [], ""
    for ch in text.replace("\n", " "):
        if PIL_FONT.getlength(cur + ch) > max_px:
            cut = cur.rfind(" ")
            if cut > len(cur) * 0.6:
                lines.append(cur[:cut])
                cur = cur[cut + 1:]
            else:
                lines.append(cur)
                cur = ""
        cur += ch
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-2].rstrip() + " …"
    return [s.strip() for s in lines]


def to_mp3(wav, mp3):
    os.makedirs(os.path.dirname(mp3), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "96k", mp3], check=True)


def main():
    spec_path, docs, lang = sys.argv[1:4]
    base = os.path.dirname(os.path.abspath(spec_path))
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    items = spec["items"]
    loaded = [read_wav(os.path.join(base, it["wav"])) for it in items]
    tmax = max(len(x) / fs for x, fs in loaded) + 0.3
    blocks = []
    for it in items:
        lines = wrap("“" + it["text"] + "”", (W_IN - 1.0) * DPI * 0.98)
        blocks.append(lines)
    # 높이(인치): 제목 0.8 + 항목마다 (머리줄 0.42 + 글 줄 0.26 + 파형 0.55 + 스펙트로그램 1.15 + 사이 0.45) + 꼬리 0.5
    item_h = [0.42 + 0.26 * len(b) + 0.55 + 1.15 + 0.45 for b in blocks]
    H = 0.8 + sum(item_h) + 0.5
    fig = plt.figure(figsize=(W_IN, H), dpi=DPI)
    fy = lambda inch: 1 - inch / H  # noqa: E731  위에서 inch 만큼 내려온 자리(그림 좌표)
    left, width = 0.5 / W_IN, (W_IN - 1.0) / W_IN
    fig.text(left, fy(0.5), spec["title"][lang], fontsize=20, fontweight="semibold", va="baseline")
    y = 0.8
    ref = max(np.percentile(20 * np.log10(np.abs(stft(x, fs=fs, nperseg=1024, noverlap=768)[2]) + 1e-9), 99.9) for x, fs in loaded)
    for it, (x, fs), lines, h in zip(items, loaded, blocks, item_h):
        fig.text(left, fy(y + 0.3), it["label"][lang], fontsize=16, fontweight="semibold", color=PURPLE, va="baseline")
        fig.text(left + width, fy(y + 0.3), it["meta"][lang], fontsize=12, color=MUTED, ha="right", va="baseline")
        y += 0.42
        for s in lines:
            fig.text(left, fy(y + 0.19), s, fontsize=13, color="#e8e8e8", va="baseline")
            y += 0.26
        ax = fig.add_axes([left, fy(y + 0.55), width, 0.55 / H])
        n = 1600
        edges = np.linspace(0, len(x), n + 1).astype(int)
        lo = np.array([x[a:b].min() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
        hi = np.array([x[a:b].max() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
        ax.fill_between(edges[:-1] / fs, lo, hi, color="#e8e8e8", linewidth=0)
        ax.set_xlim(0, tmax)
        peak = max(np.abs(x).max(), 1e-3)
        ax.set_ylim(-peak * 1.05, peak * 1.05)
        ax.axis("off")
        y += 0.55
        ax = fig.add_axes([left, fy(y + 1.15), width, 1.15 / H])
        f, t, Z = stft(x, fs=fs, nperseg=1024, noverlap=768)
        D = 20 * np.log10(np.abs(Z) + 1e-9)
        ax.imshow(D, origin="lower", aspect="auto", extent=[t[0], t[-1], 0, f[-1] / 1000], cmap="magma",
                  vmin=ref - 70, vmax=ref, interpolation="bilinear")
        ax.set_xlim(0, tmax)
        ax.set_ylim(0, 8)
        ax.set_yticks([0, 4, 8])
        ax.set_ylabel("kHz")
        for s in ax.spines.values():
            s.set_visible(False)
        # 단위는 축 오른쪽 바깥에 — 마지막 눈금 숫자와 겹치지 않게
        ax.text(1.006, 0, "초" if lang == "ko" else "s", transform=ax.transAxes, ha="left", va="center", color=MUTED, fontsize=12)
        y += 1.15 + 0.45
        if it.get("name"):  # 이름이 없으면 그림만(소리는 다른 그림이 이미 만든 것)
            to_mp3(os.path.join(base, it["wav"]), os.path.join(docs, lang, "audio", it["name"] + ".mp3"))
    fig.text(left, fy(H - 0.25), spec["footer"][lang], fontsize=12, color=MUTED, va="baseline")
    out = os.path.join(docs, lang, spec.get("out", "hero.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE).save(out, optimize=True)
    print(out, os.path.getsize(out))


main()
