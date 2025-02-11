# Sklearn: ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# (1) Exploring the Dataset
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target


# (2) Selecting the Machine Learning Model: KNN Classifier
# (3) Training the Model
X = cancer.data  # Features
y = cancer.target  # Target

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the KNN model and setting the number of neighbors
knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)  # Training the KNN model using the dataset (samples + target)

# (4) Evaluating the Results
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# (5) Hyperparameter Tuning
"""
    KNN: Hyperparameter = K
        K: 1, 2, 3,....., N
        Accuracy = %A, %B, %C,....
"""
accuracy_values = []
k_values = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker="o", linestyle="-")
plt.title("Accuracy Based on K Value")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)

# %% 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Generating random feature data
X = np.sort(5 * np.random.rand(40, 1), axis=0)  # Feature
y = np.sin(X).ravel()  # Target

# plt.figure()
# plt.scatter(X, y)

# Adding noise to the target values
y[::5] += 1 * (0.5 - np.random.rand(8))

T = np.linspace(0, 5, 500)[:, np.newaxis]

# Training KNN Regressor with different weight options
for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X, y).predict(T)
    
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="green", label="Data")
    plt.plot(T, y_pred, color="blue", label="Prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor Weights = {}".format(weight))

plt.tight_layout()
plt.show()
