import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Clf = RandomForestClassifier()

Clf.fit(X_train, Y_train)

Accuracy = Clf.score(X_test, Y_test)
print("Accuracy:", Accuracy)