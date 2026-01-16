import matplotlib.pyplot as plt
import cv2
import os
import csv
import numpy as np
from skimage.feature import peak_local_max
import utils



################ GLOBAL PARAMETERS ###############

image_path = "./subdataset" # Path to the image dataset
output_path = "./output"
blur_thresh = 50    # Threshold for blur estimation TEST
noise_thresh = 10   # Threshold for noise estimation TEST
blur_motion_thresh = 20  # Threshold for motion blur estimation TEST

WINDOW = 3
HALF = WINDOW // 2
T = 0
percent = 0.05  # percentuale per thresholding nella centroid calculation

#################################################


files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png'))]
files.sort()
for i, file in enumerate(files):
    img = cv2.imread(os.path.join(image_path, file))

    if img is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    # read image in RGB and grayscale
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # image dimensions
    H, W = gray_img.shape
    threshold = np.percentile(img, 80)


    #################### IMAGE PREPROCESSING ####################

    if utils.estimate_blur(gray_img) < blur_thresh:
        print(f"{file} has athmosphereic blur")
        threshold = np.percentile(img, 90)
        percent = 0.05

    if utils.estimate_noise(gray_img) > noise_thresh:
        continue
        print(f"{file} has noise")

    elif utils.estimate_motion_blur(gray_img) > blur_motion_thresh:     #modify when function noise is ready
        continue
        print(f"{file} has motion blur")
        
    
   
   ############################ STAR DETECTION ####################

    coordinates = peak_local_max(gray_img, min_distance=5, threshold_abs=threshold)
    print(f"Rilevati {len(coordinates)} picchi nell'immagine {file}")
    #print(coordinates)


    ###################### CENTROID CALCULATION ####################

    centroid = []
    for coord in coordinates:
        y, x = float(coord[0]), float(coord[1])
        #small local neighborhood around each detected star is selected for centroid refinement
        x1 = max(0, x - HALF)
        x2 = min(W, x + HALF + 1)
        y1 = max(0, y - HALF)
        y2 = min(H, y + HALF + 1)

        patch = cv2.getRectSubPix(gray_img.astype(np.float32), (WINDOW, WINDOW), (x, y))
        #patch = patch - patch.min()
        #patch = patch / (patch.max() + 1e-8)

        T = patch.min() + percent * (patch.max() - patch.min())
        patch_comp = utils.energy_compensation(patch)
        c = utils.threshold_centroid(patch_comp, x1-HALF, y1-HALF, T=T)
        if c is not None:
            centroid.append(c)

    print(f"Image {file}: {len(centroid)} centroidi")

    if len(centroid) == 0:
        continue  # salta immagini senza centroidi

    centroids_np = np.array(centroid)

    filename = f"centroids_image_{file}.csv"
    filepath = os.path.join(output_path, filename)

    np.savetxt(
        filepath,
        centroids_np,
        delimiter=",",
        fmt="%.3f"
    )


    ################# ERRORS ########################
    filename = "centroid_" + os.path.splitext(file)[0] + ".csv"
    filepath = os.path.join("./dataset/centroid/", filename)

    gt_coords = utils.load_ground_truth(filepath)
    #print(gt_coords)
    #print (list(centroid))

    matches, n_matched, match_ratio = utils.match_centroids(gt_coords, centroid, threshold=5.0)

    print(f"Numero di corrispondenze: {n_matched}/{len(gt_coords)}")
    print(f"Percentuale di matching: {match_ratio:.2f}%")

