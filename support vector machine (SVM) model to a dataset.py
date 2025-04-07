import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/ADMIN/Desktop/ml_2/datasets/Sample Apple vs. Orange.csv")

df = df[['Fruit Type', 'Bumpy Skin', 'Thickness of Skin (mm)']]

le1 = LabelEncoder()
df['Fruit Type'] = le1.fit_transform(df['Fruit Type'])
df['Bumpy Skin'] = le1.fit_transform(df['Bumpy Skin'])

X = df[['Bumpy Skin', 'Thickness of Skin (mm)']]
y = df['Fruit Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
