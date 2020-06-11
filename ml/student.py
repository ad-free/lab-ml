import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

df = pd.read_csv("../download/student-mat.csv", sep=";")

data = df[["G1", "G2", "G3", "studytime", "failures", "absences"]]

data.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()

predict = "G3"

X = np.array(data.drop([predict], 1))  # Features
y = np.array(data[predict])				# Label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(X_train.shape)
print(X_test.shape)


lm = LinearRegression()

model = lm.fit(X_train, y_train)

y_hat = lm.score(X_test, y_test)

print(model.predict(X_test))
print(y_hat)
