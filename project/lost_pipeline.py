import matplotlib.pyplot as plt
import cv2
import os
import csv
import numpy as np
import utils



################ GLOBAL PARAMETERS ###############

image_path = "./dataset" # Path to the image dataset
output_path = "./output"
os.makedirs(output_path, exist_ok=True)
coordinates_file = os.path.join(output_path, "LOSTcoordinates.csv")
centroids_file = os.path.join(output_path, "LOSTcentroids.csv")

files = [f for f in os.listdir(image_path) if f.lower().endswith(".png")]
files.sort()

with open(coordinates_file, "w") as f_coords, open(centroids_file, "w") as f_cents:

    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(file)

        img_name = os.path.splitext(file)[0]

        ################### STAR DETECTION + CENTROIDI ###################

        stars = utils.iterative_weighted_cog(img)
        # stars = [(x, y, rx, ry, npixels), ...]

        ################### SALVATAGGIO COORDINATE ###################

        line_coords = [img_name]
        line_coords += [f"{int(x)},{int(y)}" for x, y, _, _, _ in stars]
        f_coords.write(" ".join(line_coords) + "\n")

        ################### SALVATAGGIO CENTROIDI ###################

        line_cents = [img_name]
        line_cents += [f"{x:.3f},{y:.3f}" for x, y, _, _, _ in stars]
        f_cents.write(" ".join(line_cents) + "\n")
    print("LOST Pipeline completed successfully.")