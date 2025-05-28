import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

df = pd.read_csv(r"morti\house_price_regression_dataset.csv")

print(df.head())

df.info()

x = df.drop(columns=["House_Price"])
y = df["House_Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

print(len(x_train), len(x_test), len(y_train), len(y_test))

linreg = LinearRegression()

linreg.fit(x_train, y_train)

y_pred = linreg.predict(x_test)

mae = mean_absolute_error(y_pred, y_test)
mape = mean_absolute_percentage_error(y_pred, y_test)

print(mae, mape)
