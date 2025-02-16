Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def process_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect edges using Canny
    edges = cv2.Canny(thresh, 100, 200)

    # Label grains
    labeled = label(thresh)
    regions = regionprops(labeled)

    # Calculate grain sizes
    grain_sizes = [region.equivalent_diameter for region in regions]

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(thresh, cmap='gray')
    ax[0].set_title("Thresholded Image")
    ax[1].hist(grain_sizes, bins=20, color='blue', alpha=0.7)
    ax[1].set_title("Grain Size Distribution")
    plt.show()

    print(f"Average Grain Size: {np.mean(grain_sizes):.2f} µm")

# Run the analysis on an image
process_image("microstructure_sample.jpg")
