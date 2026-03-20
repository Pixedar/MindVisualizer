"""Load MDN vector fields from metadata JSON + binary grid files."""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


def load_field(meta_path: Path, grid_bin_path: Optional[Path] = None) -> dict:
    """
    Load MDN vector grids and metadata. Supports arbitrary K mixture components.
    Returns dict with keys: G, amin, amax, mean, mus, pi, ENT, TRAIN, meta.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    G = int(meta["grid"])
    amin = np.array(meta["axis_min"], dtype=np.float32)
    amax = np.array(meta["axis_max"], dtype=np.float32)
    files = dict(meta.get("files") or {})

    def _resolve(p):
        p = Path(p)
        return p if p.exists() else (Path(meta_path).parent / p)

    def _load_xyz3(fname):
        raw = np.fromfile(_resolve(fname), dtype=np.float32)
        return raw.reshape(G, G, G, 3).astype(np.float32)

    def _load_scalar_or_channels(fname):
        raw = np.fromfile(_resolve(fname), dtype=np.float32)
        base = G * G * G
        if raw.size == base:
            return raw.reshape(G, G, G).astype(np.float32)
        if raw.size % base == 0:
            K = raw.size // base
            return raw.reshape(G, G, G, K).astype(np.float32)
        raise ValueError(f"{fname}: size {raw.size} incompatible with G^3={base}")

    # mean field
    if "mean_xyz3" in files:
        V_mean = _load_xyz3(files["mean_xyz3"])
    else:
        if grid_bin_path is None:
            grid_bin_path = Path(meta_path).with_name(
                Path(meta_path).name.replace("_meta.json", ".bin"))
        raw = np.fromfile(grid_bin_path, dtype=np.float32)
        V_mean = raw.reshape(G, G, G, 3).astype(np.float32)

    # collect mu_k components
    mu_items = []
    for k, v in files.items():
        ks = str(k).lower()
        if "xyz3" in ks and "mu" in ks:
            m = re.search(r'(\d+)(?!.*\d)', ks)
            if m:
                mu_items.append((int(m.group(1)), v))
    mu_items.sort(key=lambda kv: kv[0])
    MUS = [_load_xyz3(fname) for _, fname in mu_items]
    K_mu = len(MUS)

    # pi (mixture weights)
    PI = None
    if "pi" in files:
        PI = _load_scalar_or_channels(files["pi"])
        if PI.ndim == 3 and K_mu == 2:
            PI = np.stack([PI, 1.0 - PI], axis=-1).astype(np.float32)

    # entropy
    ENT = None
    if "entropy" in files:
        ENT = _load_scalar_or_channels(files["entropy"])
        if ENT.ndim != 3:
            raise ValueError(f"'entropy' must be (G,G,G); got {ENT.shape}")

    # training points
    train_pts = None
    tp_key = meta.get("training_points_npy")
    if tp_key:
        p = _resolve(tp_key)
        if p.exists():
            train_pts = np.load(p).astype(np.float32)

    return {
        "G": G, "amin": amin, "amax": amax,
        "mean": V_mean,
        "mus": MUS,
        "pi": PI,
        "ENT": ENT,
        "TRAIN": train_pts,
        "meta": meta,
    }


class TriLinearSampler:
    """Trilinear interpolation of 3D vector or scalar fields at arbitrary points."""

    def __init__(self, grid_xyz3, amin, amax):
        self.g = grid_xyz3.astype(np.float32)
        self.G = grid_xyz3.shape[0]
        self.amin = amin.astype(np.float32)
        self.amax = amax.astype(np.float32)
        span = self.amax - self.amin
        self.inv_span = np.where(span > 0, 1.0 / span, 0).astype(np.float32)

    def _idx(self, P):
        f = (P - self.amin) * self.inv_span * (self.G - 1)
        ix = np.floor(f[:, 0]).astype(np.int32)
        tx = (f[:, 0] - ix).astype(np.float32)
        iy = np.floor(f[:, 1]).astype(np.int32)
        ty = (f[:, 1] - iy).astype(np.float32)
        iz = np.floor(f[:, 2]).astype(np.int32)
        tz = (f[:, 2] - iz).astype(np.float32)
        ix = np.clip(ix, 0, self.G - 2)
        iy = np.clip(iy, 0, self.G - 2)
        iz = np.clip(iz, 0, self.G - 2)
        return ix, iy, iz, tx, ty, tz

    def sample_vec(self, P):
        ix, iy, iz, tx, ty, tz = self._idx(P)
        g = self.g
        v000 = g[ix, iy, iz]
        v100 = g[ix + 1, iy, iz]
        v010 = g[ix, iy + 1, iz]
        v110 = g[ix + 1, iy + 1, iz]
        v001 = g[ix, iy, iz + 1]
        v101 = g[ix + 1, iy, iz + 1]
        v011 = g[ix, iy + 1, iz + 1]
        v111 = g[ix + 1, iy + 1, iz + 1]
        vx00 = v000 * (1 - tx)[:, None] + v100 * tx[:, None]
        vx10 = v010 * (1 - tx)[:, None] + v110 * tx[:, None]
        vx01 = v001 * (1 - tx)[:, None] + v101 * tx[:, None]
        vx11 = v011 * (1 - tx)[:, None] + v111 * tx[:, None]
        vxy0 = vx00 * (1 - ty)[:, None] + vx10 * ty[:, None]
        vxy1 = vx01 * (1 - ty)[:, None] + vx11 * ty[:, None]
        out = vxy0 * (1 - tz)[:, None] + vxy1 * tz[:, None]
        return out.astype(np.float32)

    def sample_scalar(self, S, P):
        ix, iy, iz, tx, ty, tz = self._idx(P)
        s000 = S[ix, iy, iz]
        s100 = S[ix + 1, iy, iz]
        s010 = S[ix, iy + 1, iz]
        s110 = S[ix + 1, iy + 1, iz]
        s001 = S[ix, iy, iz + 1]
        s101 = S[ix + 1, iy, iz + 1]
        s011 = S[ix, iy + 1, iz + 1]
        s111 = S[ix + 1, iy + 1, iz + 1]
        sx00 = s000 * (1 - tx) + s100 * tx
        sx10 = s010 * (1 - tx) + s110 * tx
        sx01 = s001 * (1 - tx) + s101 * tx
        sx11 = s011 * (1 - tx) + s111 * tx
        return (sx00 * (1 - ty) + sx10 * ty) * (1 - tz) + (sx01 * (1 - ty) + sx11 * ty) * tz


def load_points_any(path: Path) -> np.ndarray:
    """Load point cloud from .npy, .obj, or .ply file."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.asarray(np.load(p), np.float32)
    if p.suffix.lower() == ".obj":
        pts = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line or line[0] != 'v' or (len(line) > 1 and line[1] not in (' ', '\t')):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(pts, np.float32)
    if p.suffix.lower() == ".ply":
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            header = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Invalid PLY (no end_header)")
                header.append(line.strip())
                if line.strip() == "end_header":
                    break
            nverts = 0
            for h in header:
                if h.startswith("element vertex"):
                    nverts = int(h.split()[-1])
                    break
            pts = []
            for _ in range(nverts):
                parts = f.readline().strip().split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.asarray(pts, np.float32)
    raise ValueError(f"Unsupported point format: {p.suffix}")
