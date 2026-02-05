import matplotlib.pyplot as plt
import cv2
import os
import csv
import numpy as np
from skimage.feature import peak_local_max
import project.utils as utils



################ GLOBAL PARAMETERS ###############

image_path = "./dataset" # Path to the image dataset
output_path = "./output"
os.makedirs(output_path, exist_ok=True)
coordinates_file = os.path.join(output_path, "coordinates.csv")
centroids_file = os.path.join(output_path, "centroids.csv")

blur_thresh = 200    # Threshold for blur estimation TEST
noise_thresh = 10   # Threshold for noise estimation TEST
blur_motion_thresh = 4  # Threshold for motion blur estimation TEST

WINDOW = 3
LARGE = 5
percent = 0.9 # percentuale per thresholding nella centroid calculation

#################################################

# Download and load images
url = "https://drive.google.com/uc?id=1Tz7Vd1RLngNaPPKpzqTFgjSbUREstFHI"
files = utils.download_and_load_images(url)

files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png'))]
files.sort()

with open(coordinates_file, "w") as f_coords, open(centroids_file, "w") as f_cents:

    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(image_path, file))
        if img is None:
            raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(img, 90)


        #################### IMAGE PREPROCESSING ####################
        
        if utils.estimate_noise(gray_img) > noise_thresh:
            denoised_img = utils.denoise_image(gray_img)
            if utils.estimate_blur(denoised_img) < blur_motion_thresh:
                eccentricity_map = utils.eccentric_map(denoised_img)
                ecc_nonzero = eccentricity_map[eccentricity_map > 0]
                hist_counts, hist_bins = np.histogram(ecc_nonzero, bins=100)
                peak_bin_idx = np.argmax(hist_counts)
                ecc_mode = (hist_bins[peak_bin_idx] + hist_bins[peak_bin_idx + 1]) / 2
                coordinates = utils.detect_stars_multiscale_adaptive(denoised_img, adapt_eccentricity=True, ecc_threshold=ecc_mode, 
                                                    th_std=ecc_mode/2)
            else:
                coordinates = utils.detect_stars_multiscale(denoised_img)
                
        else:
            if utils.estimate_blur(gray_img) < blur_thresh:
                coordinates = peak_local_max(gray_img, min_distance=8, threshold_abs=np.percentile(gray_img, 80))
                coordinates = np.array([(c[1], c[0]) for c in coordinates])  
            else:
                eccentricity_map = utils.eccentric_map(denoised_img)
                ecc_nonzero = eccentricity_map[eccentricity_map > 0]
                hist_counts, hist_bins = np.histogram(ecc_nonzero, bins=100)
                peak_bin_idx = np.argmax(hist_counts)
                ecc_mode = (hist_bins[peak_bin_idx] + hist_bins[peak_bin_idx + 1]) / 2
                coordinates = utils.detect_stars_multiscale_adaptive(denoised_img, adapt_eccentricity=True, ecc_threshold=ecc_mode, 
                                                th_std=ecc_mode/2)
                print(f"Image {i} processed on {len(files)}")

        img_name = os.path.splitext(file)[0]
        line_coords = [img_name]
        line_coords += [f"{int(x)},{int(y)}" for x, y in coordinates]
        f_coords.write(" ".join(line_coords) + "\n")


        ###################### CENTROID CALCULATION ####################

        centroids_np = utils.refined_centroids(gray_img, coordinates, WINDOW=WINDOW, LARGE = LARGE)

        line_cents = [img_name]
        line_cents += [f"{x:.3f},{y:.3f}" for x, y in centroids_np]
        f_cents.write(" ".join(line_cents) + "\n") 
