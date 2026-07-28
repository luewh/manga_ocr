"""
bubble_detector.py

Detects speech/thought/narration bubbles in a manga page image and
classifies them by shape. This is a classical-CV heuristic detector
(no ML weights needed, works offline) tuned for typical manga bubble
conventions:

  - "speech"     : smooth, roughly round/oval outline        -> normal dialogue
  - "thought"    : cloud/scalloped outline (many soft bumps) -> internal monologue
  - "shout"      : jagged/spiky star outline                 -> yelling, impact
  - "narration"  : rectangular box, usually near panel edges -> captions/narration

SFX (sound effects) are deliberately NOT detected here: SFX text is
almost always drawn directly on top of the artwork with no enclosing
bubble, so by only extracting text from *inside detected bubble
contours* we exclude SFX by construction rather than by trying to
classify it.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class Bubble:
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    bubble_type: str
    confidence: float  # heuristic confidence in the shape classification


# Tunable parameters. Defaults are reasonable for ~1600-2400px tall
# scanned/digital page images; scale-relative thresholds are used
# wherever possible so this degrades gracefully at other resolutions.
class DetectorConfig:
    min_area_frac = 0.003    # ignore contours smaller than this fraction of page area
    max_area_frac = 0.35     # ignore contours larger than this (likely whole panel/page)
    min_dim_px = 70          # ignore contours smaller than this in either dimension
                              # (filters out letter counter-holes like the inside of "O"/"D"
                              # in SFX/typography, which would otherwise look bubble-like)
    solidity_speech_min = 0.92
    circularity_speech_min = 0.70
    rect_solidity_min = 0.90
    rect_angle_tol_deg = 15
    shout_norm_depth = 0.10  # convexity defect depth / sqrt(area) -- spiky vs shallow-scalloped
    dedup_iou_thresh = 0.6   # suppress nested/duplicate contours (outline has inner+outer edge)


def _order_rect_angles_ok(approx: np.ndarray, tol_deg: float) -> bool:
    """Check if a 4-point polygon has ~90 degree interior angles."""
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) != 4:
        return False
    angles = []
    for i in range(4):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    return all(abs(a - 90) <= tol_deg for a in angles)


# Below this per-defect depth (normalized by sqrt(contour area)), a convexity
# defect is treated as rasterization/anti-aliasing noise rather than an
# intentional bump -- even a perfectly smooth ellipse has dozens of sub-pixel
# "defects" from how the curve is rendered onto a pixel grid.
_SIGNIFICANT_DEFECT_NORM_DEPTH = 0.006


def _convexity_defect_stats(contour: np.ndarray) -> Tuple[int, float]:
    """
    Return (significant_defect_count, max_normalized_depth), counting only
    defects whose depth (relative to shape scale) is large enough to be an
    intentional outline feature -- e.g. a thought-bubble scallop or a
    shout-bubble spike -- rather than pixel-level curve noise.
    """
    area = cv2.contourArea(contour)
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4 or area <= 0:
        return 0, 0.0
    hull_idx = np.sort(hull_idx, axis=0)
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return 0, 0.0
    if defects is None:
        return 0, 0.0
    # cv2.convexityDefects normally returns shape (n, 1, 4), but some
    # OpenCV builds/bindings return (n, 4) directly -- normalize either way.
    defects = np.asarray(defects).reshape(-1, 4)
    depths_px = defects[:, 3] / 256.0
    norm_depths = depths_px / (np.sqrt(area) + 1e-6)
    significant = norm_depths[norm_depths > _SIGNIFICANT_DEFECT_NORM_DEPTH]
    if len(significant) == 0:
        return 0, 0.0
    return len(significant), float(significant.max())


def classify_shape(contour: np.ndarray, cfg: DetectorConfig) -> Tuple[str, float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return "unknown", 0.0

    circularity = 4 * np.pi * area / (perimeter ** 2)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    n_vertices = len(approx)

    defect_count, max_norm_depth = _convexity_defect_stats(contour)

    # Rectangle / narration box: 4-5 vertices, high solidity, ~right angles
    if n_vertices in (4, 5) and solidity >= cfg.rect_solidity_min:
        if n_vertices == 4 and _order_rect_angles_ok(approx, cfg.rect_angle_tol_deg):
            return "narration", 0.9
        if solidity >= 0.95:
            return "narration", 0.7

    # Outline has real (non-noise) bumps -> either a soft cloud (thought) or
    # sharp spikes (shout). max_norm_depth distinguishes them: shout spikes are
    # an order of magnitude deeper, relative to shape size, than thought scallops.
    if defect_count >= 3:
        if max_norm_depth >= cfg.shout_norm_depth:
            return "shout", min(1.0, max_norm_depth)
        else:
            return "thought", min(1.0, defect_count / 8)

    # Smooth round/oval, no significant bumps: classic speech bubble
    if circularity >= cfg.circularity_speech_min and solidity >= cfg.solidity_speech_min:
        return "speech", min(1.0, circularity)

    # Fallback: moderately round -> treat as speech (most common bubble type)
    if circularity >= 0.5 and solidity >= 0.85:
        return "speech", 0.4

    return "unknown", 0.2


def detect_bubbles(image: np.ndarray, cfg: DetectorConfig = DetectorConfig()) -> List[Bubble]:
    """
    image: BGR page image (as loaded by cv2.imread)
    Returns list of Bubble objects, sorted top-to-bottom then right-to-left
    (standard manga reading order for a single page; adjust for your layout).
    """
    h, w = image.shape[:2]
    page_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bubbles are typically white/flat-fill with a dark outline, so Otsu
    # threshold on the whole page separates bubble interiors well.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        frac = area / page_area
        if frac < cfg.min_area_frac or frac > cfg.max_area_frac:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        # discard small blobs: not just slivers, but also letterform counter-holes
        # (e.g. the inside of "O"/"D"/"R" in bold SFX text) which are small but can
        # be roundish enough to otherwise pass the shape checks below.
        if bw < cfg.min_dim_px or bh < cfg.min_dim_px:
            continue

        bubble_type, conf = classify_shape(contour, cfg)
        if bubble_type == "unknown":
            continue  # skip ambiguous blobs rather than mislabel

        candidates.append(Bubble(id=i, bbox=(x, y, bw, bh), contour=contour,
                                  bubble_type=bubble_type, confidence=conf))

    # Each drawn outline has both an inner and outer edge, which Otsu+CCOMP
    # picks up as two near-identical nested contours. Greedily de-duplicate:
    # keep the larger-area candidate and drop anything that overlaps it heavily.
    def iou(a, b):
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0

    candidates.sort(key=lambda b: cv2.contourArea(b.contour), reverse=True)
    kept: List[Bubble] = []
    for cand in candidates:
        if any(iou(cand, k) > cfg.dedup_iou_thresh for k in kept):
            continue
        kept.append(cand)

    # sort: top row first, right-to-left within a row (typical manga reading order)
    kept.sort(key=lambda b: (b.bbox[1] // 100, -b.bbox[0]))
    for new_id, b in enumerate(kept):
        b.id = new_id
    return kept


def find_local_bubble_contour(image: np.ndarray, bbox: Tuple[int, int, int, int],
                                padding: int = 15, cfg: DetectorConfig = DetectorConfig()) -> Bubble:
    """
    Given an approximate bubble location (e.g. from an external detector like
    comic_detector.ComicDetector), find the actual bubble outline contour
    within a small local crop and classify its shape.

    This is far more reliable than page-wide detection: the crop is tightly
    bounded around a location already known to contain a bubble, so a local
    Otsu threshold cleanly separates the bubble's outline from its immediate
    surroundings without the whole-page false-positive issues (giant panels,
    textured backgrounds) that page-wide detection runs into.

    Falls back to a plain rectangle "speech" bubble (using the given bbox
    directly as the contour) if no clean outline is found in the crop --
    this can happen with borderless bubbles or unusual art, and OCR/masking
    still works fine against the rectangle, just without a shaped legend.
    """
    x, y, w, h = bbox
    H, W = image.shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(W, x + w + padding), min(H, y + h + padding)
    crop = image[y0:y1, x0:x1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    crop_area = (x1 - x0) * (y1 - y0)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * crop_area or area > 0.98 * crop_area:
            continue  # too small to be the bubble, or basically the whole crop
        cx, cy, cw, ch = cv2.boundingRect(c)
        # a real local bubble outline should roughly fill the crop (since the
        # crop was sized to it), not hug one corner
        if cw < 0.4 * (x1 - x0) or ch < 0.4 * (y1 - y0):
            continue
        if area > best_area:
            best_area = area
            best = c

    if best is None:
        # Fallback: treat the given bbox as a plain rectangle bubble
        rect_contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
        return Bubble(id=0, bbox=(x, y, w, h), contour=rect_contour, bubble_type="speech", confidence=0.3)

    bubble_type, conf = classify_shape(best, cfg)
    if bubble_type == "unknown":
        bubble_type, conf = "speech", 0.3
    # shift contour back to full-page coordinates
    shifted = best + [x0, y0]
    bx, by, bw, bh = cv2.boundingRect(shifted)
    return Bubble(id=0, bbox=(bx, by, bw, bh), contour=shifted, bubble_type=bubble_type, confidence=conf)


LEGEND_COLORS = {
    "speech":    (60, 180, 60),    # green
    "thought":   (200, 130, 0),    # blue-ish (BGR)
    "shout":     (0, 0, 220),      # red
    "narration": (200, 0, 200),    # magenta
}
