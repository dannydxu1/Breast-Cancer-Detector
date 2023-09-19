import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV

#preprocessing
image_path = 'pgm_files'
csv_file = 'data.csv'
metadata = pd.read_csv(csv_file)
metadata.fillna(0, inplace=True)
image_size = (224, 224)

images = []
labels = []

for index, row in metadata.iterrows():
    img_file = os.path.join(image_path, str(row['REFNUM']) + '.pgm')
    if os.path.exists(img_file):
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, image_size)
        image_normalized = image_resized / 255.0
        image_flattened = image_normalized.flatten()
        images.append(image_flattened)
        label = 1 if row['SEVERITY'] == 'M' else 0
        labels.append(label)
    else:
        print(f"File not found: {img_file}")
        continue # Skip this iteration and move to t
   

images = np.array(images) 
labels = np.array(labels)


#test splitting
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
images_flatten = images.reshape(images.shape[0], -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

logistic_model = LogisticRegression(max_iter=10000)
grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test_scaled, y_test) # Checking the accuracy with the best model
print("Model accuracy with best hyperparameters:", accuracy)