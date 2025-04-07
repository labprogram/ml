import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'feature':    [1, 2, 3, 4, 5, 6, 7, 8],
    'feature1':   [2, 3, 4, 5, 6, 7, 8, 9],
    'feature2':   [3, 4, 5, 6, 7, 8, 9, 10],
    'target':     [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['feature', 'feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

threshold = 0.85
print(f"Accuracy: {acc*100:.2f}%")
if acc >= threshold:
    print(f"Model accuracy meets the desired threshold of {threshold*100:.0f}%")
else:
    print(f"Model accuracy does NOT meet the desired threshold of {threshold*100:.0f}%")
