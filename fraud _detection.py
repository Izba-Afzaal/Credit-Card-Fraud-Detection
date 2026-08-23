import pandas as pd

df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())
#fraud and genuiun transactions count
print("\nTransaction Class Count:")
print(df["Class"].value_counts())
#fraud percentage
fraud_percentage = df["Class"].mean() * 100
print("\nFraud Percentage:", fraud_percentage, "%")
print("\nAverage Transaction Amount:")

genuine_amount = df[df["Class"] == 0]["Amount"].mean()
fraud_amount = df[df["Class"] == 1]["Amount"].mean()

print("Genuine:", genuine_amount)
print("Fraud:", fraud_amount)
#genuine vs fraud transactions by graph
import matplotlib.pyplot as plt

class_counts = df["Class"].value_counts()

plt.bar(["Genuine", "Fraud"], class_counts.values)

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.yscale("log")
plt.show()
# Average Transaction Amount Graph

plt.bar(["Genuine", "Fraud"], [genuine_amount, fraud_amount])

plt.title("Average Transaction Amount")
plt.xlabel("Transaction Type")
plt.ylabel("Average Amount")

plt.show()

# Transaction Amount Distribution

plt.hist(df["Amount"], bins=50,range=(0,500))

plt.title("Transaction Amount Distribution(0,500)")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")

plt.show()
# Separate Features and Target

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
#feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)
#logistic regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nModel training completed!")
# Make predictions

y_pred = model.predict(X_test_scaled)

print("\nPredictions completed!")
#Accuracy, Precision, Recall aur F1-Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\nModel Performance:")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
#confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
plt.imshow(cm)
plt.colorbar()

plt.xticks([0, 1], ["Genuine", "Fraud"])
plt.yticks([0, 1], ["Genuine", "Fraud"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Credit Card Fraud Detection")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()
#balanced logistic regression 
from sklearn.linear_model import LogisticRegression

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(X_train_scaled, y_train)

print("\nBalanced Model Training Completed!")
# Predictions from Balanced Model

y_pred_balanced = balanced_model.predict(X_test_scaled)

print("\nBalanced Model Predictions Completed!")
#balanced model performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\nBalanced Model Performance:")

print("Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("Precision:", precision_score(y_test, y_pred_balanced))
print("Recall:", recall_score(y_test, y_pred_balanced))
print("F1 Score:", f1_score(y_test, y_pred_balanced))
#balanced model confusion matrix
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print("\nBalanced Model Confusion Matrix:")
print(cm_balanced)
#Final Fraud Prediction

sample_transaction = X_test_scaled[0].reshape(1, -1)

prediction = model.predict(sample_transaction)

if prediction[0] == 0:
    print("\nFinal Prediction: Genuine Transaction")
else:
    print("\nFinal Prediction: Fraudulent Transaction")
    import joblib

joblib.dump(balanced_model, "fraud_model.pkl")
joblib.dump(scaler, "fraud_scaler.pkl")

print("Model and scaler saved successfully!")
