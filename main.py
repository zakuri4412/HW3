import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('homedata.csv')

features = ["LotArea", "YearBuilt", "1stFlrSF", '2ndFlrSF', "FullBath", 'BedroomAbvGr', 'TotRmsAbvGrd']

target = ['SalePrice']

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

ids = range(len(y_test))

plt.figure(figsize=(12, 6))

plt.plot(ids, y_test, marker='o', linestyle='-', color='b', label='Actual')

plt.plot(ids, y_predict, marker='o', linestyle='-', color='r', label='Predicted')

plt.title('Comparison of Actual vs Predicted SalePrice')
plt.xlabel('Id of samples')
plt.ylabel('SalePrice')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()