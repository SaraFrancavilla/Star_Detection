import cv2
import csv
import os
import numpy as np
import pywt
import zipfile
import gdown
from scipy.stats import norm
from scipy.spatial.distance import cdist


def download_and_load_images(url, extract_dir=".", image_folder="dataset",
                             extensions=(".png", ".jpg", ".jpeg")):
    """
    Downloads a ZIP archive from a given URL, extracts it, and returns a sorted
    list of image filenames from the specified folder.

    :param url: Download URL of the ZIP archive.
    :param extract_dir: Directory where the ZIP archive is extracted.
    :param image_folder: Folder containing the extracted images.
    :param extensions: Allowed image file extensions.
    :return: Sorted list of image filenames.
    """
    zip_path = os.path.join(extract_dir, "dataset.zip")

    # Download silently
    print("Downloading dataset...")
    gdown.download(url, zip_path, quiet=True)

    # Extract silently
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Load image filenames
    print("Loading image list...")
    img_dir = os.path.join(extract_dir, image_folder)
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(extensions)]
    files.sort()

    print(f"{len(files)} images loaded.")
    return files

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


def align_image(img):
    img_safe = img[img >= 10] #safe_lower
    mu, sigma = norm.fit(img_safe)
    delta = 25.5 - mu  #denoised_peak
    aligned_img = np.clip(img + delta, 0, 255)
    return sigma,aligned_img

def denoise_image(img, wave='db2', level=3, k=3):

    sigma, aligned_img = align_image(img)

    # Wavelet decomposition
    coeffs = pywt.wavedec2(aligned_img, wave, level=level)

    # Threshold based on noise
    T = k * sigma

    # Process detail coefficients (skip approximation coeffs[0])
    coeffs_thresh = [coeffs[0]]  # keep the approximation untouched

    for detail_level in coeffs[1:]:
        cH, cV, cD = detail_level
        cH = pywt.threshold(cH, T, mode='soft')
        cV = pywt.threshold(cV, T, mode='soft')
        cD = pywt.threshold(cD, T, mode='soft')
        coeffs_thresh.append((cH, cV, cD))

    # Reconstruct denoised image
    denoised = pywt.waverec2(coeffs_thresh, wave)
    return np.clip(denoised, 0, 255).astype(np.uint8)


"""
Rilevamento stelle multi-scala ADATTIVO per stelle allungate.
Rileva l'eccentricità locale e adatta le finestre per stelle allungate.
    adapt_eccentricity: se True, adatta finestre per stelle allungate (img6, img7)
    ecc_threshold: soglia di eccentricità (default 0.4). Aumenta per più conservativo (0.5-0.6), diminuisci per più stelle (0.3)
    scale_eccentricity_factor: moltiplicatore (default 0.5). Aumenta a 1.0 o 1.5 per finestre più grandi
"""
def detect_stars_multiscale_adaptive(img, 
                                     A_sizes=[3,5,7], 
                                     B_sizes=[7,11,15],
                                     th_mean=3, th_std=2,
                                     adapt_eccentricity=True,
                                     ecc_threshold=0.4,
                                     scale_eccentricity_factor=0.5):

    stars = []
    H, W = img.shape

    # Computing eccentricity map if adaptive is active
    eccentricity_map = None
    if adapt_eccentricity:
        eccentricity_map = np.zeros((H, W))
        # Use a radius of 5 pixels to calculate eccentricity
        for y in range(5, H - 5):
            for x in range(5, W - 5):
                # Extract 11x11 window centered
                window = img[y-5:y+6, x-5:x+6]
                # Calculate second order moments
                yy, xx = np.meshgrid(np.arange(11), np.arange(11), indexing='ij')
                weights = window / (np.sum(window) + 1e-8)
                
                mxx = np.sum(weights * (xx - 5) ** 2)
                myy = np.sum(weights * (yy - 5) ** 2)
                mxy = np.sum(weights * (xx - 5) * (yy - 5))
                
                # Eigenvalues of the covariance matrix
                trace = mxx + myy
                det = mxx * myy - mxy ** 2
                if trace > 0:
                    lambda1 = (trace + np.sqrt(max(trace**2 - 4*det, 0))) / 2
                    lambda2 = (trace - np.sqrt(max(trace**2 - 4*det, 0))) / 2
                    # Eccentricity: (λ1 - λ2) / (λ1 + λ2)
                    if lambda1 + lambda2 > 1e-8:
                        eccentricity_map[y, x] = (lambda1 - lambda2) / (lambda1 + lambda2)

    # Loop over all scales
    for A_size, B_size in zip(A_sizes, B_sizes):
        rA = A_size // 2
        rB = B_size // 2

        for y in range(rB, H - rB):
            for x in range(rB, W - rB):

                # Adapt windows if the star is elongated
                rA_adapted = rA
                rB_adapted = rB
                
                if adapt_eccentricity and eccentricity_map is not None:
                    ecc = eccentricity_map[y, x]
                    if ecc > ecc_threshold:  # eccentricity threshold for elongation
                        # Increase search radius for elongated stars
                        scale_factor = 1 + scale_eccentricity_factor * ecc
                        rA_adapted = max(rA, int(rA * scale_factor))
                        rB_adapted = max(rB, int(rB * scale_factor))
                
                # Check bounds
                if y - rB_adapted < 0 or y + rB_adapted >= H or x - rB_adapted < 0 or x + rB_adapted >= W:
                    continue

                A = img[y-rA_adapted:y+rA_adapted+1, x-rA_adapted:x+rA_adapted+1]
                B = img[y-rB_adapted:y+rB_adapted+1, x-rB_adapted:x+rB_adapted+1]

                center = img[y, x]

                # Condition 1: the center must be a local maximum in A
                if center != np.max(A):
                    continue

                # Condition 2: the mean of A must be significantly higher than B
                mean_A = np.mean(A)
                mean_B = np.mean(B)
                if mean_A <= mean_B + th_mean:
                    continue

                # Condition 3: the variance of A must be higher than B
                var_A = np.var(A)
                var_B = np.var(B)
                if var_A <= var_B + th_std:
                    continue

                stars.append((x, y))

    # Remove duplicates keeping only the star closest to the center
    if len(stars) == 0:
        return []
    
    stars_array = np.array(stars)
    
    # Calculate distances between all stars
    distances = cdist(stars_array, stars_array)
    
    # Find clusters of nearby stars (within 3 pixels)
    keep = []
    used = set()
    
    for i in range(len(stars)):
        if i in used:
            continue
        
        # Find all nearby stars
        close_stars = np.where((distances[i] < 3) & (distances[i] > 0))[0]
        
        if len(close_stars) == 0:
            keep.append(stars[i])
            used.add(i)
        else:
            # Keep only one star per cluster
            cluster_indices = [i] + list(close_stars)
            cluster_stars = [stars[idx] for idx in cluster_indices]
            
            cluster_center = np.mean(cluster_stars, axis=0)
            distances_to_center = [np.linalg.norm(np.array(s) - cluster_center) for s in cluster_stars]
            best_idx = cluster_indices[np.argmin(distances_to_center)]
            
            keep.append(stars[best_idx])
            used.update(cluster_indices)
    
    return keep

def detect_stars_multiscale(img, 
                            A_sizes=[3,5,7], 
                            B_sizes=[7,11,15],
                            th_mean=4, th_std=3): 

    stars = []
    H, W = img.shape

    # Loop su tutte le scale
    for A_size, B_size in zip(A_sizes, B_sizes):
        rA = A_size // 2
        rB = B_size // 2

        for y in range(rB, H - rB):
            for x in range(rB, W - rB):

                A = img[y-rA:y+rA+1, x-rA:x+rA+1]
                B = img[y-rB:y+rB+1, x-rB:x+rB+1]

                center = img[y, x]

                # Condition 1: the center must be a local maximum in A
                if center != np.max(A):
                    continue

                # Condition 2: the mean of A must be significantly higher than B
                mean_A = np.mean(A)
                mean_B = np.mean(B)
                if mean_A <= mean_B + th_mean:
                    continue

                # Condition 3: the variance of A must be higher than B (star = variation)
                var_A = np.var(A)
                var_B = np.var(B)
                if var_A <= var_B + th_std:
                    continue

                stars.append((x, y))

    # Remove duplicates keeping only the star closest to the center
    # if there are multiple stars within 3 pixels
    if len(stars) == 0:
        return []
    
    stars_array = np.array(stars)
    
    # Compute pairwise distances
    distances = cdist(stars_array, stars_array)
    
    # Find clusters of nearby stars (within 3 pixels)
    keep = []
    used = set()
    
    for i in range(len(stars)):
        if i in used:
            continue
        
        # Find all nearby stars
        close_stars = np.where((distances[i] < 3) & (distances[i] > 0))[0]
        
        if len(close_stars) == 0:
            keep.append(stars[i])
            used.add(i)
        else:
            # Keep only one star per cluster (the most central one)
            cluster_indices = [i] + list(close_stars)
            cluster_stars = [stars[idx] for idx in cluster_indices]
            
            # Keep the star at the center of the cluster
            cluster_center = np.mean(cluster_stars, axis=0)
            distances_to_center = [np.linalg.norm(np.array(s) - cluster_center) for s in cluster_stars]
            best_idx = cluster_indices[np.argmin(distances_to_center)]
            
            keep.append(stars[best_idx])
            used.update(cluster_indices)
    
    return keep


def eccentric_map(debug_img):
    H, W = debug_img.shape
    eccentricity_map = np.zeros((H, W))

    # computing eccentricity (same function logic)
    for y in range(5, H - 5):
        for x in range(5, W - 5):
            window = debug_img[y-5:y+6, x-5:x+6]
            yy, xx = np.meshgrid(np.arange(11), np.arange(11), indexing='ij')
            weights = window / (np.sum(window) + 1e-8)
            
            mxx = np.sum(weights * (xx - 5) ** 2)
            myy = np.sum(weights * (yy - 5) ** 2)
            mxy = np.sum(weights * (xx - 5) * (yy - 5))
            
            trace = mxx + myy
            det = mxx * myy - mxy ** 2
            if trace > 0:
                lambda1 = (trace + np.sqrt(max(trace**2 - 4*det, 0))) / 2
                lambda2 = (trace - np.sqrt(max(trace**2 - 4*det, 0))) / 2
                if lambda1 + lambda2 > 1e-8:
                    eccentricity_map[y, x] = (lambda1 - lambda2) / (lambda1 + lambda2)
    return eccentricity_map




############# LOST FILE HANDLING FUNCTIONS #############

def basic_threshold(image):

    total_pixels = image.size
    mean = np.mean(image)
    std = np.std(image)
    return mean + (std * 5)

def iwcog_helper(image, start_idx, cutoff, checked_indices, image_width, image_height):

    star_indices = []
    x_min = start_idx % image_width
    x_max = start_idx % image_width
    y_min = start_idx // image_width
    y_max = start_idx // image_width
    max_intensity = 0
    guess = start_idx
    is_valid = True
    
    # Stack per evitare ricorsione profonda (problema in Python)
    stack = [start_idx]
    
    while stack:
        i = stack.pop()
        
        # Controlla bounds e se già visitato
        if i < 0 or i >= image_width * image_height:
            continue
        if i in checked_indices:
            continue
        if image.flat[i] < cutoff:
            continue
            
        # Marca come visitato
        checked_indices.add(i)
        star_indices.append(i)
        
        # Controlla se tocca i bordi (stella non valida)
        x = i % image_width
        y = i // image_width
        if x == 0 or x == image_width - 1 or y == 0 or y == image_height - 1:
            is_valid = False
        
        # Aggiorna bounding box
        if x > x_max:
            x_max = x
        elif x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        elif y < y_min:
            y_min = y
        
        # Aggiorna intensità massima
        if image.flat[i] > max_intensity:
            max_intensity = image.flat[i]
            guess = i
        
        # Aggiungi vicini allo stack (4-connectivity)
        if x < image_width - 1:
            stack.append(i + 1)  # right
        if x > 0:
            stack.append(i - 1)  # left
        if y < image_height - 1:
            stack.append(i + image_width)  # down
        if y > 0:
            stack.append(i - image_width)  # up
    
    return star_indices, x_min, x_max, y_min, y_max, max_intensity, guess, is_valid

def iterative_weighted_cog(image, min_change=0.0002, max_iterations=100000):

    image_height, image_width = image.shape
    result = []
    
    # Calcola la soglia
    cutoff = basic_threshold(image)
    
    # Set di pixel già controllati
    checked_indices = set()
    
    # Scansiona l'immagine pixel per pixel
    for i in range(image_height * image_width):
        # Controlla se il pixel è sopra soglia e non ancora visitato
        if image.flat[i] >= cutoff and i not in checked_indices:
            # Trova tutti i pixel connessi di questa stella
            star_indices, x_min, x_max, y_min, y_max, max_intensity, guess, is_valid = \
                iwcog_helper(image, i, cutoff, checked_indices, image_width, image_height)
            
            # Salta se stella non valida (tocca i bordi)
            if not is_valid:
                continue
            
            # Calcola dimensioni
            x_diameter = (x_max - x_min) + 1
            y_diameter = (y_max - y_min) + 1
            
            # Calcola FWHM (Full Width Half Maximum)
            count = sum(1 for idx in star_indices if image.flat[idx] > max_intensity / 2)
            fwhm = np.sqrt(count)
            
            # Calcola standard deviation dalla FWHM
            # sigma = FWHM / (2 * sqrt(2 * ln(2)))
            standard_deviation = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            modified_std_dev = 2.0 * (standard_deviation ** 2)
            
            # Prima stima: pixel con intensità massima
            guess_x_coord = float(guess % image_width)
            guess_y_coord = float(guess // image_width)
            
            # Iterazione per raffinare il centroide
            change = np.inf
            iteration = 0
            
            while change > min_change and iteration < max_iterations:
                iteration += 1
                
                # Reset delle somme pesate
                x_weighted_coord_mag_sum = 0.0
                y_weighted_coord_mag_sum = 0.0
                weighted_mag_sum = 0.0
                
                # Calcola il centroide pesato
                for idx in star_indices:
                    curr_x_coord = float(idx % image_width)
                    curr_y_coord = float(idx // image_width)
                    
                    # Calcola peso gaussiano
                    # w = I_max * exp(-((x-x_guess)^2 + (y-y_guess)^2) / (2*sigma^2))
                    exponent = -((curr_x_coord - guess_x_coord) ** 2 + 
                                 (curr_y_coord - guess_y_coord) ** 2) / modified_std_dev
                    w = max_intensity * np.exp(exponent)
                    
                    # Accumula somme pesate
                    pixel_intensity = float(image.flat[idx])
                    x_weighted_coord_mag_sum += w * curr_x_coord * pixel_intensity
                    y_weighted_coord_mag_sum += w * curr_y_coord * pixel_intensity
                    weighted_mag_sum += w * pixel_intensity
                
                # Nuova stima del centroide
                x_temp = x_weighted_coord_mag_sum / weighted_mag_sum
                y_temp = y_weighted_coord_mag_sum / weighted_mag_sum
                
                # Calcola quanto è cambiato
                change = abs(guess_x_coord - x_temp) + abs(guess_y_coord - y_temp)
                
                # Aggiorna per prossima iterazione
                guess_x_coord = x_temp
                guess_y_coord = y_temp
            
            # Aggiungi stella al risultato
            # +0.5 per centrare nel pixel (convenzione LOST)
            result.append((
                guess_x_coord + 0.5,
                guess_y_coord + 0.5,
                x_diameter / 2.0,  # radius_x
                y_diameter / 2.0,  # radius_y
                len(star_indices)  # numero di pixel
            ))
    
    return result


def match_with_ground_truth(detected_array, gt_array, max_distance=3):
    """
    Conta quante stelle rilevate matchano con il ground truth
    """
    if len(detected_array) == 0 or len(gt_array) == 0:
        return 0, 0, 0
    
    # Calcola distanze
    distances = cdist(detected_array, gt_array)
    
    # Conta matches (stella rilevata entro max_distance da una stella vera)
    matches = 0
    matched_gt = set()
    
    for i in range(len(detected_array)):
        min_dist_idx = np.argmin(distances[i])
        if distances[i, min_dist_idx] <= max_distance:
            matches += 1
            matched_gt.add(min_dist_idx)
    
    true_positives = matches
    false_positives = len(detected_array) - matches
    false_negatives = len(gt_array) - len(matched_gt)
    
    return true_positives, false_positives, false_negatives


def energy_compensation_robust(patch):
    """
    Compensazione della finestra secondo principio del paper.
    Migliora la distribuzione dell'energia per centroiding.
    """
    flat = patch.flatten()
    idx = np.argsort(flat)  # indici per ordinare

    G = flat[idx]
    G_var = np.var(G)

    n = len(G)
    # compensazione: pixel bassi aumentano, pixel alti diminuiscono
    num_extremes = max(1, n // 4)  # circa 25% estremi, regolabile
    G[:num_extremes] += G_var
    G[-num_extremes:] -= G_var

    compensated = np.zeros_like(flat)
    compensated[idx] = G
    return compensated.reshape(patch.shape)

def threshold_centroid_2(patch, x1, y1, T=0):
    """
    Calcolo centroide pesato usando threshold.
    """
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    weights = patch - T
    weights[weights < 0] = 0
    if np.sum(weights) == 0:
        return None
    xc = np.sum((xx + x1) * weights) / np.sum(weights)
    yc = np.sum((yy + y1) * weights) / np.sum(weights)
    return xc, yc

def extract_around(img, x, y, size):
    """
    Estrae patch quadrato centrato su (x,y) di dimensione size x size.
    """
    return cv2.getRectSubPix(img.astype(np.float32), (size, size), (x, y))

def refined_centroids(gray_img, coordinates, WINDOW=3, LARGE=5, percent=0.2):
    """
    Ciclo principale di estrazione centroidi con compensazione robusta.
    """
    H, W = gray_img.shape
    HALF = WINDOW // 2
    HALF_L = LARGE // 2
    centroids = []

    for coord in coordinates:
        x, y = float(coord[0]), float(coord[1])

        # patch grande per compensazione
        patchL = extract_around(gray_img, x, y, LARGE)
        patchL -= patchL.min()
        patchL /= patchL.max() + 1e-8

        patch_comp = energy_compensation_robust(patchL)

        # threshold locale basato su percentuale
        T = patch_comp.min() + percent * (patch_comp.max() - patch_comp.min())

        # calcolo centroide pesato sulla patch compensata
        c_large = threshold_centroid_2(patch_comp, x - HALF_L, y - HALF_L, T=T)
        if c_large is None:
            continue

        # optional: raffinamento locale su patch piccola
        patchS = extract_around(gray_img, c_large[0], c_large[1], WINDOW)
        patchS -= patchS.min()
        patchS /= patchS.max() + 1e-8
        T_small = patchS.min() + percent * (patchS.max() - patchS.min())
        c_small = threshold_centroid_2(patchS, c_large[0] - HALF, c_large[1] - HALF, T=T_small)
        if c_small is None:
            centroids.append(c_large)
        else:
            centroids.append(c_small)

    # ---- fusione doppioni vicini (distanza < 1.5 pixel) ----
    final = []
    used = set()
    for i in range(len(centroids)):
        if i in used:
            continue
        xi, yi = centroids[i]
        cluster = [(xi, yi)]
        used.add(i)
        for j in range(i+1, len(centroids)):
            if j in used:
                continue
            xj, yj = centroids[j]
            if np.hypot(xi-xj, yi-yj) < 1.5:  # distanza soglia per unire
                cluster.append((xj, yj))
                used.add(j)
        xs, ys = zip(*cluster)
        final.append((np.mean(xs), np.mean(ys)))

    return final

# Funzione per leggere il CSV nel formato AAA0 x1,y1 x2,y2 ... con filtro
def read_custom_csv(csv_path, suffix_char=None):
    data_dict = {}
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist.")
        return data_dict
    with open(csv_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        img_name = parts[0]

        # Filter
        if suffix_char is not None and not img_name.endswith(str(suffix_char)):
            continue

        coords = []
        for pair in parts[1:]:
            x_str, y_str = pair.split(",")
            coords.append((float(x_str), float(y_str)))
        data_dict[img_name] = np.array(coords)

    return data_dict