from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _auto_rotate_deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate small skew using Hough lines and rotate to correct.

    Returns rotated image and angle in degrees. Positive angle rotates counterclockwise.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for rho, theta in lines[:, 0]:
            ang = (theta * 180 / np.pi) - 90
            if -20 <= ang <= 20:
                angles.append(ang)
        if angles:
            angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return gray, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)


def _median_blur(gray: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(gray, 3)


def _auto_crop_margins(gray: np.ndarray) -> np.ndarray:
    """Crop uniform margins using contour of main content."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Invert for typical dark text on light background
    inverted = 255 - thresh
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    h_img, w_img = gray.shape
    # Expand a bit to avoid clipping
    x = max(0, x - 5)
    y = max(0, y - 5)
    w = min(w_img - x, w + 10)
    h = min(h_img - y, h + 10)
    return gray[y : y + h, x : x + w]


def _perspective_correction(gray: np.ndarray) -> np.ndarray:
    """Attempt simple perspective correction using largest contour's minAreaRect.
    Heuristic; only applies correction for moderate tilt.
    """
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Compute a simple upright bounding rect warp if tilt is noticeable
    angle = rect[2]
    if -10 < angle < 10:
        return gray
    w = int(rect[1][0])
    h = int(rect[1][1])
    if w <= 0 or h <= 0:
        return gray
    src_pts = box.astype(np.float32)
    dst_pts = np.array([[0, h], [0, 0], [w, 0], [w, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(gray, M, (w, h))
    return warped


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for OCR.

    Steps: grayscale -> CLAHE -> denoise -> deskew -> adaptive threshold -> auto-crop -> perspective correction -> resize up if width < 1024.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = _clahe(gray)
    gray = _median_blur(gray)
    gray, angle = _auto_rotate_deskew(gray)
    bin_img = _adaptive_binarize(gray)
    cropped = _auto_crop_margins(bin_img)
    corrected = _perspective_correction(cropped)

    h, w = corrected.shape
    if w < 1024:
        scale = 1024 / float(w)
        corrected = cv2.resize(corrected, (1024, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        logger.info("Resized image to width 1024 for OCR")

    return corrected


def preprocess_debug(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Return intermediate images for debugging visualization."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    out: Dict[str, np.ndarray] = {"grayscale": gray}
    c = _clahe(gray)
    out["clahe"] = c
    m = _median_blur(c)
    out["median_blur"] = m
    r, angle = _auto_rotate_deskew(m)
    out["deskew"] = r
    b = _adaptive_binarize(r)
    out["adaptive_thresh"] = b
    crop = _auto_crop_margins(b)
    out["auto_crop"] = crop
    persp = _perspective_correction(crop)
    out["perspective"] = persp
    h, w = persp.shape
    if w < 1024:
        scale = 1024 / float(w)
        resized = cv2.resize(persp, (1024, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        out["resized"] = resized
    else:
        out["resized"] = persp
    return out
