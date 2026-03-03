import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load dataset
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()

# Encode categorical columns
le = LabelEncoder()
penguins['species'] = le.fit_transform(penguins['species'])
penguins['sex'] = le.fit_transform(penguins['sex'])
penguins['island'] = le.fit_transform(penguins['island'])

X = penguins.drop("species", axis=1)
y = penguins["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print("Precision:", precision)
print("AUC Score:", auc)