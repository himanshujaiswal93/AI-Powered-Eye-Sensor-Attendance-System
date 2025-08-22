import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IrisTemplate:
    code: np.ndarray   # bool array of bits (H x W x K flattened)
    mask: np.ndarray   # bool array (True = valid bit)

def _estimate_pupil_center(gray: np.ndarray) -> Tuple[int, int]:
    """Rough pupil center via threshold + contours; fall back to image center."""
    h, w = gray.shape
    # Threshold to get dark pupil-ish region
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 5)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return w // 2, h // 2
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    x = int(np.clip(x, 0, w - 1)); y = int(np.clip(y, 0, h - 1))
    return x, y

def _unwrap_iris(gray: np.ndarray, out_size=(64, 256)) -> np.ndarray:
    """Polar unwrapping around pupil center; returns normalized strip."""
    cx, cy = _estimate_pupil_center(gray)
    h, w = gray.shape
    # radius from pupil to near-limbus (outer iris), conservative
    max_r = int(min(w, h) * 0.45)
    # warp to polar (angle along width, radius along height)
    polar = cv2.warpPolar(
        gray,
        (out_size[1], out_size[0]),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
    )
    # columns = angle 0..2pi; rows = radius (pupil->limbus)
    return polar

def _enhance(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _gabor_bank(kernels=(
    # (ksize, sigma, theta, lambd, gamma, psi)
    (21, 4.0, 0.0,        8.0, 0.5, 0.0),
    (21, 4.0, np.pi/4,    8.0, 0.5, 0.0),
    (21, 4.0, np.pi/2,    8.0, 0.5, 0.0),
    (21, 4.0, 3*np.pi/4,  8.0, 0.5, 0.0),
)):
    bank = []
    for ksize, sigma, theta, lambd, gamma, psi in kernels:
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        # zero-mean
        kern -= kern.mean()
        bank.append(kern)
    return bank

def _make_mask(norm: np.ndarray) -> np.ndarray:
    """Valid-bit mask: drop saturated/dark or extremely flat regions."""
    # remove extremes
    valid = (norm > 25) & (norm < 230)
    # remove low-variance pixels (flat)
    k = cv2.Laplacian(norm, cv2.CV_32F)
    valid &= (np.abs(k) > 1.0)
    return valid.astype(bool)

def create_iris_code(image_path: str,
                     unwrap_size=(64, 256)) -> IrisTemplate:
    """
    Load an eye/iris image -> normalize -> filter bank -> binary code + mask.
    Returns IrisTemplate(code, mask) where code/mask are 1D bool arrays.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    norm = _unwrap_iris(_enhance(gray), out_size=unwrap_size)
    mask2d = _make_mask(norm)

    # Gabor filtering
    bank = _gabor_bank()
    feats = []
    for kern in bank:
        resp = cv2.filter2D(norm.astype(np.float32), -1, kern)
        # binarize by sign (>=0 -> 1), gives a stable phase-like code
        bits = resp >= 0
        feats.append(bits)

    code3d = np.stack(feats, axis=-1)  # H x W x K (bool)
    mask3d = np.repeat(mask2d[:, :, None], code3d.shape[-1], axis=2)

    # Flatten for storage
    code = code3d.flatten()
    mask = mask3d.flatten()
    return IrisTemplate(code=code.astype(bool), mask=mask.astype(bool))

def compare_iris_codes(t1: IrisTemplate,
                       t2: IrisTemplate,
                       rotations: int = 8) -> Tuple[float, int]:
    """
    Hamming distance with circular shifts to tolerate slight gaze rotation.
    Returns (best_distance, best_shift).
    """
    code1, mask1 = t1.code, t1.mask
    code2, mask2 = t2.code, t2.mask
    assert code1.shape == code2.shape and mask1.shape == mask2.shape

    n = code1.size
    best_hd, best_shift = 1.0, 0
    valid_all = mask1 & mask2

    if valid_all.sum() == 0:
        return 1.0, 0  # nothing to compare

    # Try small circular shifts on code2 to align angular misregistration
    for s in range(-rotations, rotations + 1):
        if s == 0:
            c2s = code2
            m2s = mask2
        else:
            c2s = np.roll(code2, s)
            m2s = np.roll(mask2, s)

        valid = mask1 & m2s
        if valid.sum() < 50:  # not enough bits
            continue

        xor = np.logical_xor(code1[valid], c2s[valid])
        hd = xor.mean()  # Hamming distance on valid bits

        if hd < best_hd:
            best_hd, best_shift = hd, s

    return float(best_hd), int(best_shift)

def is_match(hamming_distance: float, threshold: float = 0.35) -> bool:
    """Return True if two templates are considered the same iris."""
    return hamming_distance <= threshold

# --- Simple CLI demo ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IrisCode generator & matcher")
    ap.add_argument("--img1", required=True, help="Path to first eye/iris image")
    ap.add_argument("--img2", required=True, help="Path to second eye/iris image")
    ap.add_argument("--th", type=float, default=0.35, help="Match threshold (Hamming)")
    args = ap.parse_args()

    t1 = create_iris_code(args.img1)
    t2 = create_iris_code(args.img2)
    hd, shift = compare_iris_codes(t1, t2)
    print(f"Hamming distance: {hd:.4f} (best shift {shift})")
    print("Match:", "YES ✅" if is_match(hd, args.th) else "NO ❌")
