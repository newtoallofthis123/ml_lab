import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\sravy\Desktop\class work\3-1 ML lab\Datasets\PlayTennis.csv')
print(df.head())

number = LabelEncoder()
df['Outlook'] = number.fit_transform(df['Outlook'])
df['Temperature'] = number.fit_transform(df['Temperature'])
df['Humidity'] = number.fit_transform(df['Humidity'])
df['Wind'] = number.fit_transform(df['Wind'])
df['Play Tennis'] = number.fit_transform(df['Play Tennis'])

features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"
print(df.head())

features_train, features_test, target_train, target_test = train_test_split(df[features],
df[target], test_size = 0.33, random_state = 54)

model = GaussianNB()
model.fit(features_train, target_train)

pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(accuracy)
