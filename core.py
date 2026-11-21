"""
core.py - satellite imagery, plot detection, exports and storage.
No Streamlit imports here, so everything can be tested on its own.

How detection works (all in the *same* Web-Mercator pixel grid as the map tiles,
so detected outlines line up with what the auditor sees on the satellite map):

1. Download the Esri World Imagery tiles that cover the drawn polygon and stitch them.
2. Cut the image into small colour-homogeneous superpixels (SLIC).
3. Merge neighbouring superpixels that look alike AND are not separated by a visible
   edge (field bund, track, canal) -> one region per field.
4. Keep only regions that look agricultural (green crop or bare/fallow soil), have a
   field-like size and shape, and lie inside the drawn polygon.
5. Convert region outlines back to longitude/latitude and compute areas.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from scipy import ndimage as ndi
from skimage import graph, segmentation

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TILE = 256
TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
HEADERS = {"User-Agent": "SugarMillPlotMapper/1.0 (field audit tool)"}

MAX_ZOOM = 18          # ~0.5 m/pixel in Pakistan
MIN_ZOOM = 16          # ~2 m/pixel
MAX_SIDE_PX = 2800     # largest imagery size (pixels) we are willing to download per side
PROC_MAX_PX = 2000     # image is down-scaled to this size before segmentation

M2_PER_ACRE = 4046.8564224
M2_PER_KANAL = M2_PER_ACRE / 8.0
M2_PER_HA = 10_000.0

VEG_T = 0.035          # excess-green threshold separating green crop from bare soil
EDGE_K = 0.6           # weight of visible edges when deciding whether to merge regions
MIN_SOLIDITY = 0.72    # rejects ragged / L-shaped / leaked regions


class AreaTooLarge(ValueError):
    pass


class ImageryUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Web-Mercator helpers
# --------------------------------------------------------------------------- #
def lonlat_to_px(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = TILE * (2 ** z)
    lat = max(min(lat, 85.0511287798), -85.0511287798)
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def px_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = TILE * (2 ** z)
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def metres_per_px(lat: float, z: int) -> float:
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2 ** z)


def ring_area_m2(ring_lonlat: list[tuple[float, float]]) -> float:
    """Ground area of a lon/lat ring (local equirectangular projection, fine for a few km)."""
    lat0 = sum(p[1] for p in ring_lonlat) / len(ring_lonlat)
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_574.0
    pts = [(lon * kx, lat * ky) for lon, lat in ring_lonlat]
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


# --------------------------------------------------------------------------- #
# Imagery
# --------------------------------------------------------------------------- #
@dataclass
class Imagery:
    img: np.ndarray      # H x W x 3, uint8, RGB
    z: int               # zoom level the pixels belong to
    ox: float            # global pixel x of img column 0
    oy: float            # global pixel y of img row 0
    mpp: float           # metres per pixel at the polygon's latitude
    missing: float = 0.0


def _fetch_tile(session: requests.Session, z: int, x: int, y: int):
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if arr is not None and arr.shape[:2] == (TILE, TILE):
                    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    spread = (rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)).mean()
                    if spread < 4 and rgb.mean() > 170:   # Esri's grey "no data" tile
                        return None
                    return rgb
            elif r.status_code in (400, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(0.4 * (attempt + 1))
    return None


def _fetch_mosaic(z: int, tx0: int, ty0: int, tx1: int, ty1: int):
    coords = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    session = requests.Session()
    session.headers.update(HEADERS)
    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = list(pool.map(lambda c: _fetch_tile(session, z, c[0], c[1]), coords))
    mosaic = np.full(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), 128, np.uint8)
    missing = 0
    for (x, y), t in zip(coords, tiles):
        if t is None:
            missing += 1
            continue
        r, c = (y - ty0) * TILE, (x - tx0) * TILE
        mosaic[r:r + TILE, c:c + TILE] = t
    return mosaic, missing / max(1, len(coords))


def load_imagery(ring_lonlat: list[tuple[float, float]], progress=None) -> Imagery:
    """Download imagery covering the polygon at the best zoom level that is available."""
    cb = progress or (lambda f, m: None)
    lats = [p[1] for p in ring_lonlat]
    lat_c = sum(lats) / len(lats)
    too_big = False
    for z in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        pts = [lonlat_to_px(lon, lat, z) for lon, lat in ring_lonlat]
        minx, maxx = min(p[0] for p in pts), max(p[0] for p in pts)
        miny, maxy = min(p[1] for p in pts), max(p[1] for p in pts)
        if max(maxx - minx, maxy - miny) > MAX_SIDE_PX:
            too_big = True
            continue
        pad = 12
        ox, oy = math.floor(minx) - pad, math.floor(miny) - pad
        w, h = math.ceil(maxx) + pad - ox, math.ceil(maxy) + pad - oy
        tx0, ty0 = ox // TILE, oy // TILE
        tx1, ty1 = (ox + w - 1) // TILE, (oy + h - 1) // TILE
        cb(0.05, f"Downloading satellite imagery (zoom {z})...")
        mosaic, missing = _fetch_mosaic(z, tx0, ty0, tx1, ty1)
        if missing > 0.2:
            continue                       # imagery not available at this zoom -> try a lower one
        cx, cy = ox - tx0 * TILE, oy - ty0 * TILE
        img = np.ascontiguousarray(mosaic[cy:cy + h, cx:cx + w])
        return Imagery(img, z, float(ox), float(oy), metres_per_px(lat_c, z), missing)
    if too_big and MIN_ZOOM:
        raise AreaTooLarge(
            "The drawn area is too large to analyse in one go. Draw a smaller polygon "
            "(roughly up to 5 km x 5 km) and run detection again."
        )
    raise ImageryUnavailable("Satellite imagery could not be downloaded. Check the connection and retry.")


# --------------------------------------------------------------------------- #
# Plot detection
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    min_area_acre: float = 0.5
    max_area_acre: float = 100.0
    strictness: float = 12.0        # lower = more, smaller plots; higher = merges more
    include_bare: bool = True       # also accept bare / fallow fields (not only green crop)
    min_width_m: float = 10.0       # narrower strips (roads, canals, bunds) are ignored


def _boundary_stats(labels: np.ndarray, edge: np.ndarray) -> dict:
    """Sum / count of edge strength along every boundary between two neighbouring labels."""
    k = int(labels.max()) + 1
    keys, vals = [], []
    for a, b, ea, eb in (
        (labels[:, :-1], labels[:, 1:], edge[:, :-1], edge[:, 1:]),
        (labels[:-1, :], labels[1:, :], edge[:-1, :], edge[1:, :]),
    ):
        m = a != b
        la, lb = a[m].astype(np.int64), b[m].astype(np.int64)
        keys.append(np.minimum(la, lb) * k + np.maximum(la, lb))
        vals.append(np.maximum(ea[m], eb[m]))
    keys, vals = np.concatenate(keys), np.concatenate(vals)
    if keys.size == 0:
        return {}
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    cnts = np.bincount(inv)
    return {(int(u // k), int(u % k)): (float(s), float(c)) for u, s, c in zip(uniq, sums, cnts)}


def _merge_nodes(g, src, dst):
    g.nodes[dst]["total color"] += g.nodes[src]["total color"]
    g.nodes[dst]["pixel count"] += g.nodes[src]["pixel count"]
    g.nodes[dst]["mean color"] = g.nodes[dst]["total color"] / g.nodes[dst]["pixel count"]


def _edge_weight(g, src, dst, n):
    es = ec = 0.0
    for a in (src, dst):
        if g.has_edge(a, n):
            d = g[a][n]
            es += d.get("esum", 0.0)
            ec += d.get("ecnt", 0.0)
    colour = float(np.linalg.norm(g.nodes[dst]["mean color"] - g.nodes[n]["mean color"]))
    return {"weight": colour + EDGE_K * (es / ec if ec else 0.0), "esum": es, "ecnt": ec}


def detect_plots(imagery: Imagery, ring_lonlat, params: Params, progress=None):
    """Return (features, info). Features are GeoJSON-like dicts without IDs yet."""
    cb = progress or (lambda f, m: None)
    img, z, ox, oy = imagery.img, imagery.z, imagery.ox, imagery.oy
    H, W = img.shape[:2]

    # ---- 1. working copy (down-scaled if big) --------------------------------
    s = min(1.0, PROC_MAX_PX / max(H, W))
    if s < 1.0:
        proc = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    else:
        proc = img.copy()
    ph, pw = proc.shape[:2]
    sx, sy = pw / W, ph / H
    mpp_proc = imagery.mpp / ((sx + sy) / 2.0)

    # ---- 2. polygon mask in working-copy pixels ---------------------------------
    pts = []
    for lon, lat in ring_lonlat:
        gx, gy = lonlat_to_px(lon, lat, z)
        pts.append(((gx - ox) * sx, (gy - oy) * sy))
    aoi = np.zeros((ph, pw), np.uint8)
    cv2.fillPoly(aoi, [np.round(np.array(pts) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    info = {"zoom": z, "mpp": round(imagery.mpp, 2), "regions": 0, "not_agricultural": 0,
            "too_small": 0, "too_large": 0, "bad_shape": 0}
    if int(aoi.sum()) < 2000:
        return [], info

    # ---- 3. superpixels --------------------------------------------------------
    cb(0.30, "Finding field boundaries...")
    smooth = cv2.bilateralFilter(proc, 5, 18, 5)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_s = cv2.GaussianBlur(lab, (0, 0), 1.0)
    gx_ = cv2.Sobel(lab_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_ = cv2.Sobel(lab_s, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt((gx_ ** 2 + gy_ ** 2).sum(axis=2)) / 4.0

    n_seg = int(np.clip(aoi.sum() / 900.0, 150, 3500))
    try:
        seg = segmentation.slic(smooth, n_segments=n_seg, compactness=10, sigma=0, start_label=1,
                                enforce_connectivity=True, mask=aoi.astype(bool))
    except TypeError:                                      # very old scikit-image without mask=
        seg = segmentation.slic(smooth, n_segments=n_seg, compactness=10, sigma=0, start_label=1)
        seg[aoi == 0] = 0
    seg = seg.astype(np.int64)
    seg[(aoi == 1) & (seg == 0)] = seg.max() + 1

    # ---- 4. merge look-alike neighbours that are not split by a visible edge ----
    cb(0.50, "Merging regions into plots...")
    lab_rag = lab.copy()
    lab_rag[aoi == 0] = -5000.0                            # outside polygon never merges with inside
    rag = graph.rag_mean_color(lab_rag, seg, connectivity=1)
    bstats = _boundary_stats(seg, edge)
    for u, v, d in rag.edges(data=True):
        es, ec = bstats.get((min(u, v), max(u, v)), (0.0, 0.0))
        d["esum"], d["ecnt"] = es, ec
        d["weight"] = d["weight"] + EDGE_K * (es / ec if ec else 0.0)
    merged = graph.merge_hierarchical(
        seg, rag, thresh=params.strictness, rag_copy=False, in_place_merge=True,
        merge_func=_merge_nodes, weight_func=_edge_weight,
    )
    lab_in = merged.astype(np.int64) + 1
    lab_in[aoi == 0] = 0

    # ---- 5. describe each region ----------------------------------------------
    cb(0.70, "Classifying regions...")
    n = int(lab_in.max()) + 1
    flat = lab_in.ravel()
    cnt = np.bincount(flat, minlength=n).astype(np.float64)
    safe = np.maximum(cnt, 1.0)

    def mean_of(ch):
        return np.bincount(flat, weights=ch.ravel().astype(np.float64), minlength=n) / safe

    R, G, B = (mean_of(smooth[..., i]) for i in range(3))
    gray = cv2.GaussianBlur(cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY), (0, 0), 1.2).astype(np.float64)
    m1, m2 = mean_of(gray), mean_of(gray ** 2)
    tex = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))

    tot = R + G + B + 1e-6
    r, g, b = R / tot, G / tot, B / tot
    exg = 2 * g - r - b
    bright = tot / 3.0
    green = (exg >= VEG_T) & (bright >= 35)
    bare = (params.include_bare & (exg >= -0.03) & (exg < VEG_T) & ((r - b) >= 0.03)
            & (bright >= 60) & (bright <= 215))
    agri = (green & (tex < 32)) | (bare & (tex < 18))

    px_per_m2 = 1.0 / (mpp_proc ** 2)
    min_px = params.min_area_acre * M2_PER_ACRE * px_per_m2
    max_px = params.max_area_acre * M2_PER_ACRE * px_per_m2

    present = cnt > 0
    present[0] = False
    info["regions"] = int(present.sum())
    not_agri = present & ~agri
    small = present & agri & (cnt < min_px)
    large = present & agri & (cnt > max_px)
    cand = present & agri & (cnt >= min_px) & (cnt <= max_px)
    info["not_agricultural"], info["too_small"], info["too_large"] = (
        int(not_agri.sum()), int(small.sum()), int(large.sum()))

    # ---- 6. outlines ----------------------------------------------------------
    cb(0.80, "Extracting outlines...")
    objs = ndi.find_objects(lab_in)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pad = 2
    feats = []
    for L in np.where(cand)[0]:
        if L - 1 >= len(objs) or objs[L - 1] is None:
            continue
        sl = objs[L - 1]
        sub = (lab_in[sl] == L).astype(np.uint8)
        sub = cv2.copyMakeBorder(sub, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 5:
                continue
            area_c = cv2.contourArea(c)
            area_px = area_c + len(c) / 2.0 + 1.0             # pixel count (Pick's theorem)
            if area_px < min_px or area_px > max_px:
                info["too_small" if area_px < min_px else "too_large"] += 1
                continue
            hull_a = cv2.contourArea(cv2.convexHull(c))
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            short_m, long_m = min(rw, rh) * mpp_proc, max(rw, rh) * mpp_proc
            if (hull_a <= 0 or area_c / hull_a < MIN_SOLIDITY or short_m < params.min_width_m
                    or short_m / max(long_m, 1e-6) < 0.12):
                info["bad_shape"] += 1
                continue
            eps = max(0.8, 0.003 * cv2.arcLength(c, True))
            poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(poly) < 3:
                continue
            ring = []
            for x, y in poly:
                px_p = x + sl[1].start - pad
                py_p = y + sl[0].start - pad
                lon, lat = px_to_lonlat(ox + (px_p + 0.5) / sx, oy + (py_p + 0.5) / sy, z)
                ring.append([round(lon, 7), round(lat, 7)])
            ring.append(ring[0])
            M = cv2.moments(c)
            cx_p = (M["m10"] / M["m00"] if M["m00"] else float(poly[:, 0].mean())) + sl[1].start - pad
            cy_p = (M["m01"] / M["m00"] if M["m00"] else float(poly[:, 1].mean())) + sl[0].start - pad
            c_lon, c_lat = px_to_lonlat(ox + (cx_p + 0.5) / sx, oy + (cy_p + 0.5) / sy, z)
            area_m2 = (area_px / (sx * sy)) * metres_per_px(c_lat, z) ** 2
            feats.append({
                "type": "Feature",
                "properties": {
                    "area_acre": round(area_m2 / M2_PER_ACRE, 3),
                    "area_kanal": round(area_m2 / M2_PER_KANAL, 2),
                    "area_ha": round(area_m2 / M2_PER_HA, 3),
                    "land_cover": "Green crop" if green[L] else "Bare / fallow",
                    "centroid_lat": round(c_lat, 6),
                    "centroid_lon": round(c_lon, 6),
                    "_sort": (int(cy_p // 160), float(cx_p)),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })

    feats.sort(key=lambda f: f["properties"]["_sort"])
    for f in feats:
        f["properties"].pop("_sort", None)
    info["plots"] = len(feats)
    cb(1.0, "Done")
    return feats, info


def assign_ids(features: list, prefix: str, run_code: str) -> list:
    """Unique ID per plot: PREFIX-RUNCODE-0001 (run code is unique per detection run)."""
    for i, f in enumerate(features, start=1):
        pid = f"{prefix}-{run_code}-{i:04d}"
        props = f["properties"]
        f["properties"] = {"plot_id": pid, **{k: v for k, v in props.items() if k != "plot_id"}}
    return features


def new_run_code() -> str:
    return uuid.uuid4().hex[:4].upper()


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def to_geojson(features: list, ring_lonlat=None, meta: dict | None = None) -> dict:
    fc = {"type": "FeatureCollection", "features": features}
    if meta:
        fc["properties"] = meta
    if ring_lonlat:
        fc["aoi"] = {"type": "Polygon", "coordinates": [[list(p) for p in ring_lonlat]]}
    return fc


def to_rows(features: list) -> list[dict]:
    rows = []
    for f in features:
        ring = f["geometry"]["coordinates"][0]
        wkt = "POLYGON((" + ", ".join(f"{x} {y}" for x, y in ring) + "))"
        rows.append({**f["properties"], "geometry_wkt": wkt})
    return rows


# --------------------------------------------------------------------------- #
# Storage (SQLite). NOTE: Streamlit Community Cloud's disk is ephemeral - use the
# download buttons for permanent copies, or point DB_PATH at a persistent volume.
# --------------------------------------------------------------------------- #
DB_PATH = Path(__file__).parent / "data" / "audits.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS audits (
               audit_id TEXT PRIMARY KEY, saved_at TEXT, mill TEXT, auditor TEXT, notes TEXT,
               n_plots INTEGER, total_acres REAL, aoi_json TEXT, plots_json TEXT)"""
    )
    return con


def save_audit(audit_id: str, mill: str, auditor: str, notes: str, features: list, ring_lonlat) -> None:
    total = round(sum(f["properties"]["area_acre"] for f in features), 3)
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO audits VALUES (?,?,?,?,?,?,?,?,?)",
            (audit_id, datetime.now().strftime("%Y-%m-%d %H:%M"), mill, auditor, notes,
             len(features), total, json.dumps([list(p) for p in ring_lonlat]), json.dumps(features)),
        )


def list_audits() -> list[dict]:
    with _conn() as con:
        cur = con.execute(
            "SELECT audit_id, saved_at, mill, auditor, n_plots, total_acres, notes FROM audits "
            "ORDER BY saved_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_audit(audit_id: str):
    with _conn() as con:
        row = con.execute(
            "SELECT mill, auditor, notes, aoi_json, plots_json FROM audits WHERE audit_id=?",
            (audit_id,)).fetchone()
    if not row:
        return None
    return {"mill": row[0], "auditor": row[1], "notes": row[2],
            "ring": [tuple(p) for p in json.loads(row[3])], "features": json.loads(row[4])}


def delete_audit(audit_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM audits WHERE audit_id=?", (audit_id,))
