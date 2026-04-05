import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel(r"C:\Users\LENOVO\OneDrive\Desktop\New folder (2)\airbnb_realistic_dataset.xlsx")

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

X = df[['minimum_nights',
        'number_of_reviews',
        'reviews_per_month',
        'availability_365',
        'rating',
        'host_is_superhost']]

y = df['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
joblib.dump(X.columns, "model_columns.pkl")
model.fit(X_train, y_train)

import joblib
joblib.dump(model, "airbnb_price_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

y_pred = model.predict(X_test)

print("\n📊 Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n🔝 Top Features:")
print(importance.sort_values(ascending=False))


print("\n✅ Model saved successfully!")

sample = [[2, 100, 3.5, 200, 4.5, 1]]
prediction = model.predict(sample)

print("\n Sample Prediction:")
print("Predicted Price:", round(prediction[0], 2))