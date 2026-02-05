# Star Detection and Centroid Localization in Noisy Systems

This repository contains a project focused on **star detection and centroid localization** in images affected by high levels of **electrical and luminous noise**.  
The work builds upon existing methods in the literature and aims to improve the robustness and accuracy of star tracking pipelines, with particular reference to small-satellite applications.

## Project Overview

Star trackers are critical sensors for spacecraft attitude determination. However, their performance degrades significantly in the presence of:
- electrical (read) noise  
- luminous and stray light noise  
- dark current  
- motion-induced blur and elongated star shapes  

This project combines and extends two complementary approaches from the literature:
- star detection techniques based on image processing and statistical analysis  
- centroid localization methods designed to improve accuracy under strong noise conditions  

The final objective is to enhance the detection and localization performance of the open-source **LOST** star tracking software.

## References

The project is inspired by the following works:

- *Stars Detection and Localisation Improvement Based on Image Processing*  
  Imène Taleb, Azzedine Rachedi, Khadra Benahmed  

- *High-Precision Centroid Localization Algorithm for Star Sensor Under Strong Straylight Condition*  
  Jindong Yuan, Junfeng Wu, Guohua Kang  


## Methodology

The pipeline implemented in this project follows these main steps:

1. **Dataset generation**
   - Synthetic star images generated using LOST
   - Controlled simulation of noise sources
   - Ground truth available for evaluation

2. **Noise analysis**
   - Grey-level histograms and cumulative distributions
   - Identification of background, Gaussian read noise, and dark current

3. **Image preprocessing**
   - Background alignment
   - Gaussian noise estimation
   - Wavelet-based denoising

4. **Star detection**
   - Nested-window detection strategy
   - Multi-scale analysis using different window sizes
   - Adaptive detection based on star eccentricity to handle elongated stars

5. **Performance evaluation**
   - Precision, recall, and F1-score computed against ground truth
   - Comparison between standard and adaptive detection strategies


## Repository Structure
 ```bash
├── Star_Tracking_final.ipynb # Main notebook containing the full analysis
├── pipeline.py # Detection and processing pipeline
├── lost_pipeline.py # Integration with LOST framework
├── utils.py # Utility functions
├── dataset_creation/ # Scripts for dataset generation
├── cover.png # Project cover image
├── output.zip # Example outputs
└── Star_images/ # Synthetic star image dataset
 ```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install required dependencies:
   ```bash
   pip install numpy scipy matplotlib scikit-image opencv-python pywavelets gdown
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Star_Tracking_final.ipynb
   ```

4. Run the notebook cells sequentially to reproduce the analysis and results.

## Results

* High precision in star detection, even under strong noise conditions

* Improved robustness through multi-scale and adaptive detection

* Adaptive eccentricity-based windows significantly improve detection of elongated stars

The results demonstrate that combining denoising, statistical modeling, and adaptive detection strategies leads to more reliable star tracking in challenging environments.

## Authors
Project developed as part of an academic study on star tracking and image processing by Sara Francavilla and Andreea Pollastri.