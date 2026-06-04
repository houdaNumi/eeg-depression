import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# charger la data
features = np.load("../data/features.npy")
labels = np.load("../data/labels.npy")

print(features.shape)
print(labels.shape)

# 1. split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 2. feature selection
selector = SelectKBest(f_classif, k=6)
x_train_selected = selector.fit_transform(x_train, y_train)
x_test_selected = selector.transform(x_test)

# 3. normalisation
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_selected)
x_test_scaled = scaler.transform(x_test_selected)

# 4. modèle
model = SVC(kernel='rbf', random_state=42)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))