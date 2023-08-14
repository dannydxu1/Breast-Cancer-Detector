import numpy as np
import pandas as pd
import cv2
import os

# Define the path to your images and CSV file
image_path = 'pgm_files'
csv_file = 'data.csv'

# Read the metadata from the CSV file
metadata = pd.read_csv(csv_file)
metadata.fillna(0, inplace=True)

# Define the desired image size
image_size = (224, 224)

# Create lists to hold the processed images and labels
images = []
labels = []

for index, row in metadata.iterrows():
    # Build the path to the corresponding image
    img_file = os.path.join(image_path, row['REFNUM'] + '.pgm') # .pgm extension
    
    # Read and resize the image
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, image_size)
    
    # Normalize the image pixel values
    image_normalized = image_resized / 255.0
    
    # Append the processed image to the images list
    images.append(image_normalized)
    
    # Map the severity to a numerical label (you can customize this mapping)
    label = 1 if row['SEVERITY'] == 'M' else 0
    labels.append(label)

# Convert lists to numpy arrays
images = np.array(images).reshape(-1, image_size[0], image_size[1], 1) # 1 channel for grayscale
labels = np.array(labels)

# Your preprocessed images and labels are now in the 'images' and 'labels' arrays
