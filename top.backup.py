import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.naive_bayes import GaussianNB

# Baca gambar uji
image = cv2.imread('F:/kuliah/Skripsi/detect_app_flask/dataset/testing/7.jpg')

# Tentukan ukuran cropping
crop_size = 100

# Crop image dari tengah
height, width, _ = image.shape
start_h = (height - crop_size) // 2
start_w = (width - crop_size) // 2
cropped_image = image[start_h:start_h+crop_size, start_w:start_w+crop_size]

# Konversi gambar ke ruang warna HSI
hsi_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
h, s, i = cv2.split(hsi_img)

# Hitung nilai rata-rata saluran H, S, dan I
h_mean = np.mean(h)
s_mean = np.mean(s)
i_mean = np.mean(i)

# convert image to grayscale
gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
# calculate GLCM matrix and extract features
glcm = graycomatrix(gray_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
contrast = graycoprops(glcm, 'contrast').mean()
correlation = graycoprops(glcm, 'correlation').mean()
energy = graycoprops(glcm, 'energy').mean()
homogeneity = graycoprops(glcm, 'homogeneity').mean()
dissimilarity = graycoprops(glcm, 'dissimilarity').mean()

# Menggabungkan fitur-fitur menjadi satu vektor fitur
feature_vector = np.array([h_mean, s_mean, i_mean, contrast, correlation, energy, homogeneity, dissimilarity])

# Membaca file CSV menggunakan pandas
data = pd.read_csv('F:/kuliah/Skripsi/detect_app_flask/training_data.csv')

# Memisahkan fitur (X_train) dan label (y_train)
X_train = data[['H_mean','S_mean','I_mean','Contrast','Correlation','Energy','Homogeneity','Dissimilarity']]  # Menggunakan kolom-kolom yang ditentukan sebagai label
y_train = data[['Label']]  # Menggunakan kolom-kolom yang ditentukan sebagai label

# Menampilkan hasil
# print("Fitur (X_train):")
# print(X_train[:40])  

# print("\nLabel (y_train):")
# print(y_train[:40])  


# Mengambil 8 fitur dari vektor fitur gambar uji
feature_vector = feature_vector[:8]

# Inisialisasi model Naive Bayes
model = GaussianNB()

# Latih model dengan data training
model.fit(X_train, y_train)

# Prediksi kelas untuk gambar uji
y_pred = model.predict([feature_vector])

# Print hasil prediksi
print("Hasil prediksi:", y_pred)
