import pandas as pd

# importing dataset

df = pd.read_csv("datasets/cardata.csv")

print(df.head())

# exporting data

from pandas import DataFrame

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
'Price': [32000,35000,37000,45000]
}

df = DataFrame(Cars, columns= ['Brand', 'Price'])

exported = df.to_excel("datasets/cardata.xlsx", header=True)
