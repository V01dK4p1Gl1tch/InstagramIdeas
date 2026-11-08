#!/usr/bin/env python3
"""
photo_ascii.py - turn a photo into an animated "letter art" video (9:16)
(optionally with a live-typing code panel underneath: --panel).

    python photo_ascii.py my_photo.jpg                      # -> my_photo_ascii.mp4
    python photo_ascii.py my_photo.jpg --preview 8          # one PNG frame at t=8s
    python photo_ascii.py cat.jpg --eyes "0.38,0.42,0.05;0.62,0.42,0.05" --focus 0.5,0.4

Needs: numpy, opencv-python, pillow, ffmpeg (in PATH).
"""
import argparse
import math
import os
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920         # output size (9:16); changed by --size
CW, CH = 5, 9             # size of one letter cell, px; changed by --cell
ART_H = H                 # height of the letter picture (smaller when --panel)
COLS, ROWS = W // CW, ART_H // CH
CODE_FONT_SIZE = 13
SCALE = 1.0               # W / 720, keeps text sizes sensible


def set_layout(width, cw, ch, panel):
    """Sets the global layout. Called once from main()."""
    global W, H, CW, CH, ART_H, COLS, ROWS, CODE_FONT_SIZE, SCALE
    W, H = width, int(round(width * 16 / 9 / 2)) * 2
    CW, CH = cw, ch
    SCALE = W / 720.0
    CODE_FONT_SIZE = int(round(13 * SCALE))
    ART_H = int(H * 0.65) if panel else H
    COLS, ROWS = W // CW, ART_H // CH
    ART_H = ROWS * CH

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def find_font(user_font=None):
    names = [user_font] if user_font else []
    names += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "DejaVuSansMono.ttf",
    ]
    for n in names:
        if n and (os.path.exists(n) or os.path.basename(n) == n):
            try:
                ImageFont.truetype(n, 10)
                return n
            except OSError:
                continue
    return None


def get_font(path, size):
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


# --------------------------------------------------------------------------
# 1. photo -> subject mask
# --------------------------------------------------------------------------
def cut_subject(img_bgr, mask_path=None, no_cut=False):
    """Returns float mask 0..1 (same size as image)."""
    h, w = img_bgr.shape[:2]
    if no_cut:
        return np.ones((h, w), np.float32)
    if mask_path:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (w, h))
        return (m.astype(np.float32) / 255.0)
    # automatic GrabCut: assume the subject is roughly in the centre
    s = 480.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * s), int(h * s)))
    sh, sw = small.shape[:2]
    rect = (int(sw * 0.04), int(sh * 0.03), int(sw * 0.92), int(sh * 0.94))
    gm = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(small, gm, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    m = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return m.astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# 2. letter atlas: every glyph sorted by how much "ink" it has
# --------------------------------------------------------------------------
def build_atlas(font_path, charset, text=None):
    font = get_font(font_path, CH)
    chars = list(charset)
    glyphs = []
    for ch in chars:
        im = Image.new("L", (CW, CH), 0)
        ImageDraw.Draw(im).text((0, -1), ch, fill=255, font=font)
        glyphs.append(np.asarray(im, np.float32) / 255.0)
    ink = np.array([g.mean() for g in glyphs])
    order = np.argsort(ink)
    atlas = np.stack([glyphs[i] for i in order])       # dark -> bright
    atlas = np.concatenate([np.zeros((1, CH, CW), np.float32), atlas])  # 0 = empty
    return atlas, [chars[i] for i in order]


# --------------------------------------------------------------------------
# 3. the "living" part: wind field, blink, camera
# --------------------------------------------------------------------------
class Scene:
    def __init__(self, img_bgr, mask, args):
        # work resolution: keep it modest, the output is only 120x83 cells
        big = max(img_bgr.shape[:2])
        s = 2400.0 / big if big > 2400 else (1600.0 / big if big < 1600 else 1.0)
        if s != 1.0:
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=interp)
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
        self.scale = s                     # working px per original photo px
        self.h, self.w = img_bgr.shape[:2]
        # local contrast boost (CLAHE) so that dark areas and eyes keep their detail
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # auto-brightness: dark / foggy photos get lifted so the letters glow
        inside = mask > 0.5
        if inside.sum() > 100:
            lum0 = (self.rgb @ np.array([0.299, 0.587, 0.114], np.float32))[inside]
            gain = float(np.clip(0.80 / max(np.percentile(lum0, 96), 1e-3), 1.0, 2.4))
            self.rgb = np.clip(self.rgb * gain, 0, 1)
        # unsharp mask: crisper detail survives the trip into letters
        blur = cv2.GaussianBlur(self.rgb, (0, 0), 2.0)
        self.rgb = np.clip(self.rgb + args.sharp * (self.rgb - blur), 0, 1)
        self.mask = mask.astype(np.float32)
        self.args = args

        ys, xs = np.nonzero(self.mask > 0.5)
        if len(xs) < 100:                                # cut failed -> whole photo
            self.mask[:] = 1.0
            ys, xs = np.mgrid[0:self.h, 0:self.w]
            ys, xs = ys.ravel(), xs.ravel()
        self.bbox = (xs.min(), ys.min(), xs.max(), ys.max())

        # --- wind weight: edges of the subject (hair, cloth, fur) and the top
        #     move most, the "core" and the bottom stay anchored
        inside = (self.mask > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
        d = max(8.0, 0.10 * max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))
        edge = np.clip(1.0 - dist / d, 0, 1) * inside
        yy = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]
        top = np.clip((self.bbox[3] - np.arange(self.h)) /
                      max(1.0, self.bbox[3] - self.bbox[1]), 0, 1)[:, None] ** 1.2
        wgt = 0.30 * top + 0.70 * edge
        self.wind_w = cv2.GaussianBlur(wgt.astype(np.float32), (0, 0), d * 0.35)
        self.gx, self.gy = np.meshgrid(np.arange(self.w, dtype=np.float32),
                                       np.arange(self.h, dtype=np.float32))

        # --- blink
        self.eyes = []
        if args.eyes:
            for part in args.eyes.split(";"):
                ex, ey, er = [float(v) for v in part.split(",")]
                self.eyes.append((ex * self.w, ey * self.h, er * self.w))
        self.eye_m = np.zeros((self.h, self.w), np.float32)
        self.eye_col = np.zeros(3, np.float32)
        for ex, ey, er in self.eyes:
            cv2.ellipse(self.eye_m, (int(ex), int(ey)), (int(er * 1.2), int(er)),
                        0, 0, 360, 1.0, -1, cv2.LINE_AA)
            ring = np.zeros((self.h, self.w), np.uint8)
            cv2.circle(ring, (int(ex), int(ey)), int(er * 2.0), 1, -1)
            cv2.circle(ring, (int(ex), int(ey)), int(er * 1.3), 0, -1)
            self.eye_col += self.rgb[ring > 0].mean(0)
        if self.eyes:
            self.eye_col /= len(self.eyes)
            self.eye_m = cv2.GaussianBlur(self.eye_m, (0, 0), max(1.0, er * 0.15))

        # --- camera: starts zoomed on `focus`, pulls back to the whole subject
        x0, y0, x1, y1 = self.bbox
        bw, bh = (x1 - x0), (y1 - y0)
        asp = (COLS * CW) / (ART_H)
        self.full_c = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        self.full_h = max(bh * 1.12, bw * 1.12 / asp)
        fx, fy = [float(v) for v in args.focus.split(",")]
        self.face_c = np.array([fx * self.w, fy * self.h], np.float32)
        self.face_h = self.full_h / args.zoom

        # cell grid sampled at 4x then box-filtered (looks smoother)
        gx, gy = np.meshgrid((np.arange(COLS * 4) + 0.5) / (COLS * 4) - 0.5,
                             (np.arange(ROWS * 4) + 0.5) / (ROWS * 4) - 0.5)
        self.cell_x, self.cell_y = gx.astype(np.float32), gy.astype(np.float32)
        self.asp = asp

    # wind strength over time, 0..1 - gusts like in the original
    def wind(self, t):
        g = 0.5 + 0.5 * math.sin(0.55 * t + 0.7)
        g *= 0.65 + 0.35 * math.sin(1.9 * t)
        return float(np.clip(g, 0, 1)) * self.args.wind

    def camera(self, t):
        k = smoothstep((t - 1.2) / (self.args.duration * 0.55))
        k = k * k * (3 - 2 * k)
        c = self.face_c * (1 - k) + self.full_c * k
        hv = self.face_h * (1 - k) + self.full_h * k
        return c, hv, k

    def displacement(self, t, wnd):
        """Per-pixel motion (dx, dy) in working pixels. Override this in a
        subclass to decide what moves (see photo_wind.py)."""
        amp = wnd * 0.030 * self.full_h + 0.004 * self.full_h
        dx = amp * self.wind_w * (np.sin(1.3 * t + self.gy * 0.011 + self.gx * 0.003)
                                  + 0.6 * np.sin(3.1 * t + self.gy * 0.03 + self.gx * 0.012))
        dy = 0.45 * amp * self.wind_w * np.sin(2.3 * t + self.gx * 0.02)
        # breathing
        dy = dy + 0.004 * self.full_h * np.sin(1.7 * t) * (1 - self.wind_w) * \
            (self.gy - self.bbox[3]) / max(1.0, self.full_h) * 6
        return dx, dy

    def render_cells(self, t):
        """returns (rgb cells (ROWS,COLS,3), alpha (ROWS,COLS), zoom, wind)."""
        wnd = self.wind(t)
        dx, dy = self.displacement(t, wnd)

        c, hv, k = self.camera(t)
        wv = hv * self.asp
        sx = c[0] + self.cell_x * wv
        sy = c[1] + self.cell_y * hv
        # displacement is sampled at the same (source) place and applied there
        ix = np.clip(sx, 0, self.w - 1).astype(np.int32)
        iy = np.clip(sy, 0, self.h - 1).astype(np.int32)
        mx = (sx - dx[iy, ix]).astype(np.float32)
        my = (sy - dy[iy, ix]).astype(np.float32)

        rgb = self.rgb
        # blink
        if self.eyes:
            b = 0.0
            for t0 in self.args.blinks:
                q = (t - t0) / 0.30
                if 0 <= q <= 1:
                    b = max(b, 1 - abs(2 * q - 1))
            if b > 0:
                em = self.eye_m[..., None] * b
                rgb = rgb * (1 - em) + self.eye_col * em

        col = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = cv2.remap(self.mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        col = cv2.resize(col * a[..., None], (COLS, ROWS), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        col = col / np.maximum(a[..., None], 1e-3)
        return col, a, 1.0 / max(0.05, hv / self.full_h), wnd


def letters_frame(col, alpha, t, atlas, chars, text, shimmer):
    """cells -> big picture made of letters."""
    lum = col @ np.array([0.299, 0.587, 0.114], np.float32)
    lum = np.clip(lum, 0, 1) ** 0.7
    n = len(atlas) - 1
    # even the darkest cells keep a visible letter (shadows must not become holes)
    idx = 1 + np.minimum(((0.22 + 0.78 * lum) * n).astype(np.int32), n - 1)
    # shimmer: neighbouring glyphs of similar ink swap now and then
    if shimmer:
        rs = np.random.RandomState(int(t * 5))
        idx = np.clip(idx + rs.randint(-1, 2, idx.shape), 1, n)
    visible = alpha > 0.35
    idx = np.where(visible, idx, 0)
    if text:
        # fixed text, brightness only controls colour/opacity
        tt = np.array([ord(ch) for ch in text], np.int32)
        pos = (np.arange(ROWS)[:, None] * COLS + np.arange(COLS)[None, :]) % len(tt)
        sel = tt[pos]
        lut = {ord(ch): i + 1 for i, ch in enumerate(chars)}
        mapped = np.array([lut.get(int(v), 0) for v in sel.ravel()], np.int32).reshape(sel.shape)
        idx = np.where(visible & (mapped > 0), mapped, 0)
    glyph = atlas[idx]                                    # (R,C,CH,CW)
    tint = np.clip(col * 1.30 + 0.12, 0, 1) * (0.62 + 0.60 * lum[..., None])
    img = glyph[..., None] * tint[:, :, None, None, :]    # (R,C,CH,CW,3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(ROWS * CH, COLS * CW, 3)
    return img, int(visible.sum())


# --------------------------------------------------------------------------
# 4. code panel (types its own source, with syntax colours)
# --------------------------------------------------------------------------
KEYWORDS = set("""def return if else elif for in while import from as class not and or is
None True False with try except lambda pass break continue""".split())
TOK = re.compile(r"""(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d[\d_.]*\b)|([A-Za-z_]\w*)""")
C_BG = (14, 14, 18)
C_TEXT = (205, 208, 218)
C_COM = (92, 99, 112)
C_STR = (152, 195, 121)
C_NUM = (209, 154, 102)
C_KW = (224, 108, 117)
C_FN = (97, 175, 239)
C_CALL = (198, 120, 221)


def colorize(line):
    """-> list of (col, text, color)"""
    out, pos = [], 0
    for m in TOK.finditer(line):
        if m.start() > pos:
            out.append((pos, line[pos:m.start()], C_TEXT))
        s = m.group(0)
        if m.group(1):
            c = C_COM
        elif m.group(2):
            c = C_STR
        elif m.group(3):
            c = C_NUM
        else:
            if s in KEYWORDS:
                c = C_KW
            elif line[m.end():m.end() + 1] == "(":
                c = C_FN if line[:m.start()].rstrip().endswith("def") else C_CALL
            else:
                c = C_TEXT
        out.append((m.start(), s, c))
        pos = m.end()
    if pos < len(line):
        out.append((pos, line[pos:], C_TEXT))
    return out


class CodePanel:
    def __init__(self, font_path, source_lines, script_name):
        self.font = get_font(font_path, CODE_FONT_SIZE)
        self.small = get_font(font_path, 12)
        self.lines = [l.rstrip("\n").replace("\t", "    ") for l in source_lines]
        self.cw = self.font.getlength("M") if font_path else 7
        self.lh = CODE_FONT_SIZE + 5
        self.top = ART_H + 14
        self.nrows = int((H - self.top - 44) // self.lh)
        self.maxcols = int((W - 56) // self.cw)
        self.name = script_name
        self.cum = np.cumsum([len(l) + 1 for l in self.lines])
        self.cache = {}

    def toks(self, i):
        if i not in self.cache:
            self.cache[i] = colorize(self.lines[i][:self.maxcols])
        return self.cache[i]

    def draw(self, canvas, progress, t, fps):
        total = int(self.cum[-1])
        pos = int(progress * total)
        li = int(np.searchsorted(self.cum, pos, side="right"))
        li = min(li, len(self.lines) - 1)
        col_in_line = pos - (int(self.cum[li - 1]) if li > 0 else 0)
        first = max(0, li - self.nrows + 1)

        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        d.rectangle([0, ART_H, W, H], fill=C_BG)
        d.line([0, ART_H, W, ART_H], fill=(40, 40, 52), width=1)
        for r, i in enumerate(range(first, li + 1)):
            y = self.top + r * self.lh
            num = f"{i + 1:3d}"
            d.text((6, y), num, font=self.font, fill=(70, 74, 88))
            x0 = 46
            if i == li:
                d.rectangle([0, y - 2, W, y + self.lh - 3], fill=(40, 22, 30))
            for c0, text, color in self.toks(i):
                if i == li:
                    text = text[:max(0, col_in_line - c0)]
                if text:
                    d.text((x0 + c0 * self.cw, y), text, font=self.font, fill=color)
            if i == li:
                cx = x0 + min(col_in_line, self.maxcols) * self.cw
                if int(t * 2.2) % 2 == 0:
                    d.rectangle([cx, y, cx + self.cw * 0.8, y + self.lh - 4], fill=(240, 90, 150))
        d.text((12, H - 28), f"$ py {self.name}", font=self.small, fill=(150, 156, 170))
        d.text((W - 12 - self.small.getlength(f"running · {fps} fps"), H - 28),
               f"running · {fps} fps", font=self.small, fill=(120, 200, 140))
        return np.asarray(im)


def draw_hud(canvas, font_path, t, zoom, wind, letters, handle):
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    f = get_font(font_path, int(round(12 * SCALE)))
    k = SCALE
    d.text((W - 90 * k, 14 * k), f"t {t:05.2f}s", font=f, fill=(235, 110, 110))
    y = ART_H - 70 * k
    if handle:
        d.text((16 * k, y), f"@{handle} · 2026", font=f, fill=(150, 155, 165))
    d.text((16 * k, y + 18 * k), f"zoom {zoom:4.2f}  wind {wind:4.2f}", font=f, fill=(110, 115, 125))
    d.text((16 * k, y + 36 * k), f"Letters {letters:,}", font=f, fill=(110, 115, 125))
    return np.asarray(im).copy()


# --------------------------------------------------------------------------
# 5. main
# --------------------------------------------------------------------------
def main(scene_cls=None, mask_fn=None, defaults=None):
    """scene_cls / mask_fn / defaults let other scripts (photo_wind.py) reuse
    everything here with their own motion and cut-out."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("-o", "--out")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview", type=float, metavar="T", help="render one frame at T seconds to PNG")
    ap.add_argument("--focus", default="0.5,0.35", help="where the camera starts, as fractions x,y of the photo")
    ap.add_argument("--zoom", type=float, default=2.4, help="start zoom (1 = whole subject)")
    ap.add_argument("--wind", type=float, default=1.0, help="wind strength multiplier")
    ap.add_argument("--eyes", help='blink: "x,y,r;x,y,r" as fractions of the photo width/height')
    ap.add_argument("--blinks", default="2.5,7,12.5,17", help="blink times, seconds")
    ap.add_argument("--mask", help="your own mask image (white = subject)")
    ap.add_argument("--no-cut", action="store_true", help="don't remove the background")
    ap.add_argument("--charset", default="letters", help="'letters', 'symbols' or your own set of characters")
    ap.add_argument("--text", help="fill the picture with this text instead of ink-sorted letters")
    ap.add_argument("--handle", default="", help="your @name in the corner")
    ap.add_argument("--audio", help="music file to add")
    ap.add_argument("--font", help="path to a monospace .ttf")
    ap.add_argument("--panel", action="store_true", help="show the typing-code panel under the picture")
    ap.add_argument("--code", help="file to show in the code panel (default: this script)")
    ap.add_argument("--no-hud", action="store_true", help="hide the t / zoom / wind / Letters text")
    ap.add_argument("--size", type=int, default=1080, help="output width in px (height = 16:9 of it); 720 is faster")
    ap.add_argument("--cell", default="5x9", help="letter cell WxH in px; smaller = more detail, e.g. 4x7")
    ap.add_argument("--sharp", type=float, default=0.8, help="sharpening of the source photo")
    ap.add_argument("--crf", type=int, default=25, help="video quality, lower = better and bigger")
    if defaults:
        ap.set_defaults(**defaults)
    args = ap.parse_args()
    args.blinks = [float(x) for x in args.blinks.split(",") if x]
    cw, ch = [int(v) for v in args.cell.lower().split("x")]
    set_layout(args.size, cw, ch, args.panel)

    img = cv2.imread(args.photo)
    if img is None:
        sys.exit(f"can't read image: {args.photo}")
    font_path = find_font(args.font)
    if not font_path:
        print("warning: no monospace TTF found, use --font path/to/font.ttf", file=sys.stderr)

    print("cutting out the subject...")
    mask = mask_fn(img, args) if mask_fn else cut_subject(img, args.mask, args.no_cut)
    scene = (scene_cls or Scene)(img, mask, args)

    sets = {
        "letters": "abcdefghijklmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ0123456789",
        "symbols": ".,:;-~=+*#%@&$/\\|()[]{}<>!?",
    }
    charset = sets.get(args.charset, args.charset)
    if args.text:
        charset = "".join(sorted(set(charset + args.text.replace(" ", ""))))
    atlas, chars = build_atlas(font_path, charset)
    text = (args.text or "").replace(" ", "") or None

    panel = None
    if args.panel:
        src_path = args.code or os.path.abspath(__file__)
        with open(src_path, encoding="utf-8") as fh:
            source = fh.readlines()
        # start the typing at the first real code, skip the file header
        start = next((i for i, l in enumerate(source) if l.startswith("import ")), 0)
        panel = CodePanel(font_path, source[start:], os.path.basename(src_path))

    total = int(args.fps * args.duration)

    def frame(i):
        t = i / args.fps
        col, a, zoom, wnd = scene.render_cells(t)
        art, nl = letters_frame(col, a, t, atlas, chars, text, shimmer=True)
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:art.shape[0], :art.shape[1]] = np.clip(art * 255, 0, 255).astype(np.uint8)
        if not args.no_hud:
            canvas = draw_hud(canvas, font_path, t, zoom, wnd, nl, args.handle)
        if panel is None:
            return canvas
        prog = min(1.0, t / (args.duration * 0.97))     # typing progress 0..1
        return panel.draw(canvas, prog, t, args.fps)

    if args.preview is not None:
        i = int(args.preview * args.fps)
        out = args.out or os.path.splitext(args.photo)[0] + f"_preview_{args.preview:g}s.png"
        Image.fromarray(frame(i)).save(out)
        print("saved", out)
        return

    out = args.out or os.path.splitext(args.photo)[0] + "_ascii.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-"]
    if args.audio:
        cmd += ["-i", args.audio, "-c:a", "aac", "-b:a", "160k", "-shortest",
                "-af", f"afade=t=out:st={max(0, args.duration - 1.5)}:d=1.5"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(total):
        p.stdin.write(frame(i).tobytes())
        if i % args.fps == 0:
            print(f"\r{i / total * 100:5.1f}%", end="", flush=True)
    p.stdin.close()
    p.wait()
    print(f"\rdone -> {out}")


if __name__ == "__main__":
    main()