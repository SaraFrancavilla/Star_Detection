import cv2
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Error: missing image path", file=sys.stderr)
    sys.exit(1)

image_path = sys.argv[1]

img = cv2.imread(image_path)

if img is None:
    print(f"Error: image '{image_path}' not found or cannot be opened", file=sys.stderr)
    sys.exit(1)

# Convert to RGB and grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect only stars within certain dimensions
def detect_stars(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            stars.append((cx, cy))
    return len(stars)       #we want only the number of stars

# threshold (adjustable)
_, binary = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)
star_count = detect_stars(binary)
print(star_count)