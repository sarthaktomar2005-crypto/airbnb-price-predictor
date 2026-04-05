import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Airbnb Price Predictor", layout="centered")

st.title("🏡 Airbnb Price Prediction App")
st.markdown("developed by Sarthak Tomar ")
st.markdown("Enter listing details to predict price 💰")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "airbnb_price_model.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))


minimum_nights = st.number_input("Minimum Nights", 1, 30, 2)
number_of_reviews = st.number_input("Number of Reviews", 0, 500, 50)
reviews_per_month = st.number_input("Reviews per Month", 0.0, 10.0, 2.0)
availability_365 = st.number_input("Availability (days)", 0, 365, 100)
rating = st.number_input("Rating", 1.0, 5.0, 4.0)
host_is_superhost = st.selectbox("Superhost", [0, 1])

city = st.selectbox("City", ["Delhi", "Mumbai", "Goa", "Jaipur"])
room_type = st.selectbox("Room Type", ["Private room", "Shared room"])
season = st.selectbox("Season", ["Peak", "Off-Peak"])
input_dict = {
    'minimum_nights': minimum_nights,
    'number_of_reviews': number_of_reviews,
    'reviews_per_month': reviews_per_month,
    'availability_365': availability_365,
    'rating': rating,
    'host_is_superhost': host_is_superhost,
    'city': city,
    'room_type': room_type,
    'season': season
}
input_df = pd.DataFrame([input_dict])
input_df['availability_ratio'] = input_df['availability_365'] / 365
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

st.write("Input columns:", input_df.columns)

missing_cols = set(model_columns) - set(input_df.columns)
st.write("Missing columns:", missing_cols)

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]

    if prediction < 2000:
        st.warning(f"💸 Low Price: ₹{round(prediction,2)}")
    elif prediction < 5000:
        st.info(f"💰 Medium Price: ₹{round(prediction,2)}")
    else:
        st.success(f"🔥 High Price: ₹{round(prediction,2)}")
import matplotlib.pyplot as plt

st.subheader("📊 Feature Importance")

importance = model.feature_importances_
features = model_columns

fig, ax = plt.subplots()
ax.barh(features[:10], importance[:10])
st.pyplot(fig)
print("end to end project made by Sarthak Tomar ")