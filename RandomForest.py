from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

oli = fetch_olivetti_faces()

# show image
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(oli.images[i], cmap = "gray")
    plt.axis("off")
plt.show()   

X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 

estimators = [5,10,100,500]
accuracy_values = []

for i in range(len(estimators)):
    rf_clf = RandomForestClassifier(n_estimators = estimators[i], random_state = 42)  
    rf_clf.fit(X_train, y_train) 
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)

plt.figure()
plt.plot(estimators, accuracy_values, marker="o", linestyle="-")
plt.title("Accuracy Based on Estimators")
plt.xlabel("Estimators")
plt.ylabel("Accuracy")
plt.xticks(estimators)


# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

house = fetch_california_housing()

X = house.data
y = house.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf_rg = RandomForestRegressor(random_state = 42)

rf_rg.fit(X_train, y_train)

y_pred = rf_rg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("mse: ",mse)

rmse = root_mean_squared_error(y_test, y_pred)
print("rmse: ",rmse)

