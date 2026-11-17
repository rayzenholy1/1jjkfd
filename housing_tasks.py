import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor

housing = fetch_california_housing(as_frame=True)
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
X = housing.data[features]
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Завдання 1")
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
r2_tree = r2_score(y_test, tree.predict(X_test))
print("R2 дерева рішень на тестовій вибірці:", round(r2_tree, 4))

print("\nЗавдання 2")
forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
r2_forest = r2_score(y_test, forest.predict(X_test))
table2 = pd.DataFrame({
    "Модель": ["DecisionTreeRegressor", "RandomForestRegressor"],
    "R2 (test)": [round(r2_tree, 4), round(r2_forest, 4)]
})
print(table2.to_string(index=False))

print("\nЗавдання 3")
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
r2_xgb = r2_score(y_test, xgb.predict(X_test))
table3 = pd.DataFrame({
    "Модель": ["RandomForestRegressor", "XGBRegressor"],
    "R2 (test)": [round(r2_forest, 4), round(r2_xgb, 4)]
})
print(table3.to_string(index=False))
if r2_xgb > r2_forest:
    print("XGBoost показав кращий результат, ніж Random Forest.")
else:
    print("Random Forest показав кращий результат, ніж XGBoost.")

print("\nЗавдання 4")
median_price = np.median(y)
y_class = (y > median_price).astype(int)
print("Медіана ціни:", round(median_price, 4))

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(Xc_train, yc_train)
pred = log_model.predict(Xc_test)

cm = confusion_matrix(yc_test, pred)
print("Confusion matrix:")
print(cm)
print("Accuracy:", round(accuracy_score(yc_test, pred), 4))
print("Precision:", round(precision_score(yc_test, pred), 4))
print("Recall:", round(recall_score(yc_test, pred), 4))
print("F1:", round(f1_score(yc_test, pred), 4))

ConfusionMatrixDisplay(cm, display_labels=["0 (≤ медіани)", "1 (> медіани)"]).plot(cmap="Blues")
plt.title("Confusion matrix")
plt.show()
