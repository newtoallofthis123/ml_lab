from sklearn.preprocessing import StandardScaler, Binarizer, MinMaxScaler

import pandas as pd
import numpy as np

df = pd.read_csv("datasets/diabetes.csv")
print(df.head())


arr = df.values

X = arr[:, 0:8]
y = arr[:, 8]


# 2a StandardScaler

print("StandardScaler")

scaled = StandardScaler().fit_transform(X)

print(scaled)

# 2b Binarizer

print("Binarizer")

binarized = Binarizer(threshold=0).fit_transform(X)

print(binarized)

# 2c MinMaxScaler

minmaxed = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
print(minmaxed)
