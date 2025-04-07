import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("C:/Users/ADMIN/Desktop/ml_2/datasets/Loan_Prediction_Dataset.csv")
data.drop(columns=["Loan_ID"], inplace=True)
imputer = SimpleImputer(strategy="most_frequent")
data.iloc[:, :] = imputer.fit_transform(data)
data["Dependents"] = data["Dependents"].replace("3+", 3).astype(int)
label_encoders = {}
categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
print("Gaussian Naive Bayes model Precision(in %):", metrics.precision_score(y_test, y_pred)*100)
print("Gaussian Naive Bayes model Recall(in %):", metrics.recall_score(y_test, y_pred)*100)
print("Gaussian Naive Bayes model F1-Score(in %):", metrics.f1_score(y_test, y_pred)*100)