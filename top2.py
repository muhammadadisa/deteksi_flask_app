import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.naive_bayes import GaussianNB
from flask import Flask, jsonify, request
import base64

# Create the Flask app
app = Flask(__name__)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    image_data = request.files['image'].read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    
    # Read the image
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Crop image from the center
    crop_size = 100
    height, width, _ = image.shape
    start_h = (height - crop_size) // 2
    start_w = (width - crop_size) // 2
    cropped_image = image[start_h:start_h+crop_size, start_w:start_w+crop_size]
    
    # Convert image to HSI color space
    hsi_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    h, s, i = cv2.split(hsi_img)
    
    # Calculate mean values of H, S, and I channels
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    i_mean = np.mean(i)
    
    # Convert image to grayscale
    gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM matrix and extract features
    glcm = graycomatrix(gray_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    
    # Combine features into a feature vector
    feature_vector = np.array([h_mean, s_mean, i_mean, contrast, correlation, energy, homogeneity, dissimilarity])
    
    # Read the training data CSV file using pandas
    data = pd.read_csv('F:/kuliah/Skripsi/detect_app_flask/training_data.csv')
    
    # Separate features (X_train) and labels (y_train)
    X_train = data[['H_mean', 'S_mean', 'I_mean', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Dissimilarity']]
    y_train = data[['Label']]
    
    # Initialize the Naive Bayes model
    model = GaussianNB()
    
    # Train the model with the training data
    model.fit(X_train, y_train)
    
    # Get the first 8 features from the feature vector of the test image
    feature_vector = feature_vector[:8]
    
    # Predict the class for the test image
    y_pred = model.predict([feature_vector])
    
    # Return the prediction result and the encoded image as JSON
    return jsonify({'prediction': y_pred, 'image': encoded_image})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
