import pandas as pd
from sklearn.ensemble import RandomForestClassifier

d = {
    'f1': [5, 6, 7, 8, 5, 6, 7, 9, 4, 6, 8, 7],
    'f2': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'label': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(d)

X = df[['f1', 'f2']]
y = df['label']

m = RandomForestClassifier(n_estimators=20, oob_score=True, bootstrap=True)
m.fit(X, y)

print("OOB Error:", 1 - m.oob_score_)
