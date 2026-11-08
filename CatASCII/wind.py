#!/usr/bin/env python3
"""
photo_wind.py - the knight-cat video: the cat stands still (he only blinks),
the wind moves ONLY the cape and the grass around his feet.

    python photo_wind.py knight.jpg                 # -> knight_wind.mp4
    python photo_wind.py knight.jpg --preview 9     # one PNG frame at t=9s
    python photo_wind.py knight.jpg --debug         # saves layers.png - check the masks

It reuses everything from photo_ascii.py (keep both files in one folder),
so all of its options work here too: --size, --cell, --sharp, --audio, ...

HOW TO USE IT FOR ANOTHER PHOTO
Everything photo-specific is in the LAYERS block below, in pixels of the
ORIGINAL photo:
  KNIGHT_RECT  box around the figure (it is cut out with GrabCut)
  EXTRA_POLYS  thin parts GrabCut loses (here: the sword)
  CAPE_POLY    outline of the cloth that should flutter
  ANCHOR       where the cloth is attached (it does not move there)
  GRASS_*      ellipse of grass around the feet that should sway
Run with --debug, look at layers.png, fix the numbers, repeat.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat as pa  # noqa: E402

# ----------------------------------------------------------------------------
# LAYERS  (pixels of the original 783x391 photo)
# ----------------------------------------------------------------------------
KNIGHT_RECT = (304, 112, 442, 354)                  # x0, y0, x1, y1
KNIGHT_HULL = [(340, 150), (346, 128), (372, 119), (398, 124), (423, 141), (431, 170),
               (424, 196), (432, 215), (436, 262), (438, 300), (428, 322), (406, 349),
               (334, 350), (317, 338), (324, 300), (321, 262), (311, 232), (312, 204),
               (344, 198)]                          # "probably the knight"
KNIGHT_CORE = [(350, 140), (410, 140), (414, 185), (410, 212), (420, 280), (400, 330),
               (345, 335), (335, 280), (330, 215), (350, 195)]   # "surely the knight"
EXTRA_POLYS = [[(334, 16), (343, 58), (345, 202), (330, 202), (328, 58)]]   # the sword
CAPE_POLY = [(398, 199), (416, 195), (430, 201), (470, 222), (516, 254), (537, 270),
             (536, 284), (525, 301), (500, 318), (478, 313), (452, 303), (433, 293),
             (419, 268), (411, 238), (403, 216)]
ANCHOR = (410, 207)                                 # cape is pinned here
CAPE_LEN = 135                                      # px from the anchor to the far tip
GRASS_CENTER, GRASS_AXES = (388, 344), (128, 31)    # ellipse of swaying grass

DEFAULTS = dict(
    focus="0.485,0.405", zoom=2.9,
    eyes="0.4457,0.4360,0.0085;0.4773,0.4380,0.0085",
    blinks="2.4,6.5,11.2,15.8",
    wind=1.0,
)

STASH = {}      # sub-masks handed from the cut-out step to the scene


def poly_mask(h, w, polys, blur=0.0):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 255)
    m = m.astype(np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), blur) if blur else m


def cut_layers(img, args):
    h, w = img.shape[:2]
    # --- the knight: GrabCut seeded with hand-drawn hull / core
    gm = np.full((h, w), cv2.GC_BGD, np.uint8)
    x0, y0, x1, y1 = KNIGHT_RECT
    gm[y0:y1, x0:x1] = cv2.GC_PR_BGD
    hull = poly_mask(h, w, [KNIGHT_HULL]) > 0.5
    core = poly_mask(h, w, [KNIGHT_CORE]) > 0.5
    gm[hull & (gm != cv2.GC_BGD)] = cv2.GC_PR_FGD
    gm[core] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, gm, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    knight = ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    knight = cv2.morphologyEx(knight, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    knight = cv2.morphologyEx(knight, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(knight)
    if n > 1:                                        # keep only the biggest piece
        knight = ((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    knight = knight.astype(np.float32) / 255.0
    sword = poly_mask(h, w, EXTRA_POLYS)
    body = np.maximum(knight, sword)
    # the cape polygon is cut out of the body, so the two never fight
    cape = poly_mask(h, w, [CAPE_POLY], blur=0.8)
    body = body * (1 - cape * (poly_mask(h, w, [CAPE_POLY]) > 0.5))
    body = np.maximum(body, 0)
    # --- grass patch under the feet: soft ellipse
    grass = np.zeros((h, w), np.float32)
    cv2.ellipse(grass, GRASS_CENTER, GRASS_AXES, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    grass = cv2.GaussianBlur(grass, (0, 0), 7.0)
    grass = np.clip(grass * 1.6 - 0.25, 0, 1)
    STASH.update(body=body, cape=cape, grass=grass)
    return np.clip(np.maximum.reduce([body, cape, grass]), 0, 1)


class WindScene(pa.Scene):
    """Cat = fixed.  Cape and grass = moved by the wind."""

    def __init__(self, img, mask, args):
        super().__init__(img, mask, args)
        k = self.scale
        self.k = k
        # tighter framing: the figure should fill the frame
        x0, y0, x1, y1 = self.bbox
        self.full_h = max((y1 - y0) * 1.07, (x1 - x0) * 1.06 / self.asp)
        self.face_h = self.full_h / args.zoom

        def up(m):
            return cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        body, cape, grass = up(STASH["body"]), up(STASH["cape"]), up(STASH["grass"])
        body_soft = cv2.GaussianBlur(body, (0, 0), 2.5 * k)
        # cape weight: 0 at the shoulder, growing to the far tip
        ax, ay = ANCHOR[0] * k, ANCHOR[1] * k
        self.ax = ax
        d = np.hypot(self.gx - ax, self.gy - ay) / (CAPE_LEN * k)
        ramp = np.clip((d - 0.06) / 0.94, 0, 1) ** 1.25
        cape_zone = cv2.GaussianBlur(cv2.dilate((cape > 0.3).astype(np.float32),
                                                np.ones((int(14 * k) | 1,) * 2, np.uint8)),
                                     (0, 0), 6 * k)
        self.w_cape = ramp * cape_zone * (1 - body_soft)
        # grass weight: stronger in the front (lower in the photo)
        low = np.clip((self.gy - 295 * k) / (70 * k), 0, 1)
        self.w_grass = np.clip(grass * (0.30 + 0.70 * low), 0, 1) * (1 - body_soft)
        self.s_cape = np.clip((self.gx - ax) / (CAPE_LEN * k), 0, 1.4)
        # blade-like variation: every patch of grass sways a bit out of phase
        self.phase = (self.gx / k) * 0.055 + (self.gy / k) * 0.031

    def gust(self, t):
        # slow gusts + quick flutter; 0..1
        g = 0.5 + 0.5 * np.sin(0.62 * t + 0.4)
        g = g * (0.7 + 0.3 * np.sin(1.7 * t + 1.1))
        return float(np.clip(0.18 + 0.95 * g, 0, 1)) * min(1.6, self.args.wind)

    def displacement(self, t, wnd):
        k = self.k
        g = self.gust(t)
        tau = 2 * np.pi
        # --- cape: waves travel from the shoulder to the tip, gusts lift it
        wave1 = np.sin(tau * 0.85 * t - self.s_cape * 5.2)
        wave2 = np.sin(tau * 1.7 * t - self.s_cape * 9.0 + 1.3)
        dy_c = self.w_cape * k * ((2.6 + 4.6 * g) * wave1 + 1.1 * g * wave2 - 5.5 * g)
        dx_c = self.w_cape * k * ((1.6 + 3.4 * g) * np.sin(tau * 0.85 * t - self.s_cape * 4.4 + 1.0)
                                  + 4.0 * g)
        # --- grass: leans with the gust and trembles
        a = (0.9 + 2.2 * g) * k
        dx_g = self.w_grass * (a * np.sin(tau * 0.42 * t + self.phase)
                               + 0.45 * a * np.sin(tau * 0.95 * t + 2.3 * self.phase)
                               + 3.4 * g * k)
        dy_g = self.w_grass * 0.30 * a * np.sin(tau * 0.60 * t + 1.7 * self.phase)
        self._g = g
        return (dx_c + dx_g).astype(np.float32), (dy_c + dy_g).astype(np.float32)

    def wind(self, t):                       # number shown in the HUD
        return self.gust(t)


def debug_layers(path):
    img = cv2.imread(path)
    cut_layers(img, None)
    vis = img.astype(np.float32) * 0.45
    for key, col in (("body", (0, 255, 0)), ("cape", (255, 80, 80)), ("grass", (0, 200, 255))):
        m = STASH[key][..., None]
        vis = vis * (1 - 0.5 * m) + np.array(col, np.float32) * 0.5 * m
    cv2.circle(vis, ANCHOR, 4, (0, 0, 255), -1)
    out = os.path.splitext(path)[0] + "_layers.png"
    cv2.imwrite(out, cv2.resize(vis.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    print("saved", out, "  green = knight (still), blue = cape, yellow = grass")


if __name__ == "__main__":
    if "--debug" in sys.argv:
        debug_layers([a for a in sys.argv[1:] if not a.startswith("-")][0])
    else:
        pa.main(scene_cls=WindScene, mask_fn=cut_layers, defaults=DEFAULTS)