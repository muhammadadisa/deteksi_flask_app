import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def crop_images(image_dir, output_dir, crop_size):
    image_files = os.listdir(image_dir)
    cropped_images = []
    
    for file in image_files:
        image_path = os.path.join(image_dir, file)
        img = cv2.imread(image_path)
        
        # crop image from the center
        height, width, _ = img.shape
        start_h = (height - crop_size) // 2
        start_w = (width - crop_size) // 2
        patch = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
        cropped_images.append(patch)

        # Save cropped patches with modified file name
        original_filename = os.path.splitext(file)[0]  # Get the original file name without extension
        patch_name = f"{original_filename}.jpg"  # Compose the new file name
        patch_path = os.path.join(output_dir, patch_name)
        cv2.imwrite(patch_path, patch)


def extract_features_hsi_glcm(image_dir, csv_path):
    image_files = os.listdir(image_dir)
    features = []
    
    for file in image_files:
        image_path = os.path.join(image_dir, file)
        img = cv2.imread(image_path)
        
        # convert image to HSI color space
        hsi_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, i = cv2.split(hsi_img)
        
        # extract features from HSI channels
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        i_mean = np.mean(i)
        
        # convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # calculate GLCM matrix and extract features
        glcm = graycomatrix(gray_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        
        # Get the label based on the filename
        original_filename = os.path.splitext(file)[0]
        label = 1  # Default label (not fresh)
        if original_filename.isdigit() and 1 <= int(original_filename) <= 20:
            label = 0  # Set the label as fresh
        
        features.append([original_filename, h_mean, s_mean, i_mean, contrast, correlation, energy, homogeneity, dissimilarity, label])
    
    # save features to CSV file
    df = pd.DataFrame(features, columns=['Image', 'H_mean', 'S_mean', 'I_mean', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Dissimilarity', 'Label'])
    
    # Sort DataFrame by 'Image' column in ascending order
    df_sorted = df.sort_values(by='Image', ascending=True)
    
    # Save sorted DataFrame to CSV file
    df_sorted.to_csv(csv_path, index=False)


# Path dan parameter
image_dir = 'F:/kuliah/Skripsi/detect_app_flask/dataset/traning'
output_dir = 'F:/kuliah/Skripsi/detect_app_flask/out'
csv_path = 'F:/kuliah/Skripsi/detect_app_flask/training_data.csv'
crop_size = 100

# Crop gambar
crop_images(image_dir, output_dir, crop_size)

# Ekstraksi fitur HSI dan GLCM, lalu simpan ke file CSV
extract_features_hsi_glcm(output_dir, csv_path)
