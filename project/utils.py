import cv2
import csv
import numpy as np



def estimate_blur(img_gray):
    """
    Estimate the blur level of a grayscale image using the variance of the Laplacian.
    """
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    var_lap = lap.var()
    return var_lap

def estimate_motion_blur(img_gray):
    """
    Estimate motion blur using gradient magnitude method.
    """
    # gradient magnitude
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # gradient mean → più bassa indica blur da movimento
    blur_measure = np.mean(grad_mag)
    return blur_measure

def estimate_noise(img_gray):
    """
    Estimate the noise level of a grayscale image using a simple patch-based method.
    """
    patch = img_gray[100:200, 100:200]  #portion of image
    noise_std = np.std(patch)
    return noise_std


def detect_stars(img_gray):
    """
    Detect stars in a binary image using contour detection.
    Returns a list of star centroids.
    """ 
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            stars.append((cx, cy))
    return len(stars)

def energy_compensation(patch):
    """
    Docstring per energy_compensation
    
    :param patch: Descrizione
    """
    flat = patch.flatten()
    idx = np.argsort(flat)

    G = flat[idx]
    G_var = np.var(G)

    # compensazione
    G[0] += G_var
    G[1] += G_var
    G[-1] -= G_var
    G[-2] -= G_var

    compensated = np.zeros_like(flat)
    compensated[idx] = G

    return compensated.reshape(patch.shape)

def threshold_centroid(patch, x1, y1, T=0):
    """
    Docstring per threshold_centroid
    
    :param patch: Descrizione
    :param x1: Descrizione
    :param y1: Descrizione
    :param T: Descrizione
    """
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]

    #T = np.median(patch)
    weights = patch - T
    weights[weights < 0] = 0

    if np.sum(weights) == 0:
        return None

    xc = np.sum((xx + x1) * weights) / np.sum(weights)
    yc = np.sum((yy + y1) * weights) / np.sum(weights)

    return xc, yc 


def load_ground_truth(csv_file):
    """
    Legge un CSV in formato Lost (solo x e y) e restituisce una lista di coordinate (x, y)
    """
    coords = []
    with open(csv_file, "r") as f:
        for line in f:
            if line.startswith("num_input_centroids"):
                continue
            parts = line.split()
            key = parts[0]
            value = float(parts[1])
            if "_x" in key or "_y" in key:
                coords.append(value)

    xy_coords = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    return xy_coords

def match_centroids(ground_truth, detected, threshold=30.0):
    """
    Confronta i centroidi calcolati con quelli di ground truth.

    Args:
        ground_truth: lista di tuple (x, y)
        detected: lista di tuple (x, y) calcolati dal tuo algoritmo
        threshold: distanza massima (in pixel) per considerare una corrispondenza

    Returns:
        matches: lista di tuple ((x_calcolato, y_calcolato), (x_gt, y_gt))
        n_matched: numero di centroidi correttamente abbinati
        match_ratio: percentuale di centroidi corretti rispetto al totale di ground truth
    """
    ground_truth = np.array(ground_truth, dtype=float)
    detected = np.array(detected, dtype=float)

    matches = []
    for c in detected:
        # distanza da tutti i punti di ground truth
        distances = np.linalg.norm(ground_truth - c, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] <= threshold:
            matches.append((tuple(c), tuple(ground_truth[min_idx])))

    n_matched = len(matches)
    match_ratio = n_matched / len(ground_truth) * 100 if len(ground_truth) > 0 else 0.0

    return matches, n_matched, match_ratio
