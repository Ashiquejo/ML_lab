import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Check if the dataset exists locally; if not, download it directly from a public repository
if not os.path.exists("diabetes.csv"):
    print("Dataset not found locally. Downloading 'diabetes.csv' from GitHub...")
    # URL to the raw CSV file hosted on GitHub
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    # Column names based on the dataset description
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    data = pd.read_csv(url, names=columns)
    # Save a local copy for future use
    data.to_csv("diabetes.csv", index=False)
    print("Download complete and saved as 'diabetes.csv'.\n")
else:
    # Step 2: Load Dataset and Preprocessing
    data = pd.read_csv("diabetes.csv")
    print("Dataset loaded successfully from local file.\n")

# Separate input features (X) and target variable (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Logistic Regression Without Feature Scaling
print("--- Logistic Regression Without Feature Scaling ---")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
acc_unscaled = accuracy_score(y_test, y_pred)
prec_unscaled = precision_score(y_test, y_pred)
rec_unscaled = recall_score(y_test, y_pred)
f1_unscaled = f1_score(y_test, y_pred)

print(f"Accuracy : {acc_unscaled:.4f}")
print(f"Precision: {prec_unscaled:.4f}")
print(f"Recall   : {rec_unscaled:.4f}")
print(f"F1 Score : {f1_unscaled:.4f}\n")

# Step 4: Logistic Regression With Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- Logistic Regression With Feature Scaling ---")
model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)

# Calculate metrics
acc_scaled = accuracy_score(y_test, y_pred_scaled)
prec_scaled = precision_score(y_test, y_pred_scaled)
rec_scaled = recall_score(y_test, y_pred_scaled)
f1_scaled = f1_score(y_test, y_pred_scaled)

print(f"Accuracy : {acc_scaled:.4f}")
print(f"Precision: {prec_scaled:.4f}")
print(f"Recall   : {rec_scaled:.4f}")
print(f"F1 Score : {f1_scaled:.4f}\n")

# Step 5: Visualizing the Results
print("Generating performance comparison graph...")
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
unscaled_metrics = [acc_unscaled, prec_unscaled, rec_unscaled, f1_unscaled]
scaled_metrics = [acc_scaled, prec_scaled, rec_scaled, f1_scaled]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, unscaled_metrics, width, label='Without Scaling', color='#4C72B0')
rects2 = ax.bar(x + width/2, scaled_metrics, width, label='With Scaling', color='#DD8452')

# Add styling and labels
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison: Unscaled vs Scaled Features')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15) # Leave a bit of room at the top for the numbers
ax.legend(loc='lower right')

# Attach a text label above each bar showing the exact score
ax.bar_label(rects1, fmt='%.3f', padding=3)
ax.bar_label(rects2, fmt='%.3f', padding=3)

fig.tight_layout()
plt.show()