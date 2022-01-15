import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import pickle

data = pd.read_csv("iris.csv")

train, test = train_test_split(
    data, test_size=0.4, stratify=data["class"], random_state=42
)

X_train = train[["sepallength", "sepalwidth", "petallength", "petalwidth"]]
y_train = train[["class"]]
X_test = test[["sepallength", "sepalwidth", "petallength", "petalwidth"]]
y_test = test[["class"]]

mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)
prediction = mod_dt.predict(X_test)
print(
    "The accuracy of the Decision Tree is",
    "{:.3f}".format(metrics.accuracy_score(prediction, y_test)),
)

pickle.dump(mod_dt, open("decisiontreeclassifier.pkl", "wb"))
