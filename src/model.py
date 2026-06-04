import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


#charger la data
features = np.load("../data/features.npy")
labels = np.load("../data/labels.npy")

print(features.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)

model = KNeighborsClassifier(n_neighbors=5)  # créer le modèle
model.fit(x_train, y_train)  # entraîner sur les données train
y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))