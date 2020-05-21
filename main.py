import pandas as pd
import numpy as np
import matplotlib.pylab as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

path = "data.csv"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-door", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight",
           "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio",
           "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.read_csv(path, names=headers)

''' Replace ? to NaN'''
df.replace("?", np.nan, inplace=True)
print(df.head(5))
print(df.describe(include="all"))

# drive_wheels_counts = df["drive-wheels"].value_counts()
# print(drive_wheels_counts)
# 
# df.rename(columns={"drive-wheels": 'values_counts'}, inplace=True)
# df.index.name = 'drive-wheels'
# 
# print(df.describe(include="all"))

y = df["engine-size"]
x = df["price"]

plt.scatter(x, y)
plt.title("Scattler plot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.xlabel("Price")
plt.show()
