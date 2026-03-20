"""Colormap utilities: Turbo, Rainbow, and bipolar white-center."""

import numpy as np


def turbo_rgb01(t):
    t = np.clip(np.asarray(t, np.float32), 0, 1)
    r = (34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38345.17 + 14829.80 * t))))) / 255.0
    g = (23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * 2199.29)))) / 255.0
    b = (27.20 + t * (3211.10 + t * (-15327.97 + t * (34592.87 + t * (-30538.66 + 9347.97 * t))))) / 255.0
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def rainbow_rgb01(t):
    t = np.clip(np.asarray(t, np.float32), 0.0, 1.0)
    h = t * 4.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    q = 1.0 - f
    i = np.clip(i, 0, 3)
    zeros = np.zeros_like(t)
    ones = np.ones_like(t)
    r = np.choose(i, [zeros, zeros, f, ones])
    g = np.choose(i, [f, ones, ones, q])
    b = np.choose(i, [ones, q, zeros, zeros])
    start = (t <= 0.0)
    end = (t >= 1.0)
    if np.any(start):
        r[start], g[start], b[start] = 0.0, 0.0, 1.0
    if np.any(end):
        r[end], g[end], b[end] = 1.0, 0.0, 0.0
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def bicolor_white_center(dent_vals, pos=(255, 120, 0), neg=(0, 100, 255)):
    d = np.asarray(dent_vals, np.float32)
    a = np.abs(d)
    scale = float(np.percentile(a, 97)) if a.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    t = np.clip(a / scale, 0.0, 1.0)[:, None]
    white = np.array([[255, 255, 255]], np.float32)
    pos_c = np.array(pos, np.float32)[None, :]
    neg_c = np.array(neg, np.float32)[None, :]
    out = np.where(d[:, None] >= 0.0,
                   (1.0 - t) * white + t * pos_c,
                   (1.0 - t) * white + t * neg_c)
    return out.astype(np.uint8)
