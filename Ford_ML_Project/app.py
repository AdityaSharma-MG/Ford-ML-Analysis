"""Interactive Streamlit app for Ford used-car price exploration and prediction."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Ford Price Explorer", layout="wide")
DATA_DIR = Path(__file__).resolve().parent
REQUIRED = ["model", "year", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize", "price"]

@st.cache_data
def load_data():
    path = DATA_DIR / "ford_cleaned.csv"
    if not path.exists():
        path = DATA_DIR / "ford.csv"
    if not path.exists():
        raise FileNotFoundError("Place ford_cleaned.csv or ford.csv beside app.py.")
    data = pd.read_csv(path)
    missing = [col for col in REQUIRED if col not in data.columns]
    if missing:
        raise ValueError("Dataset is missing expected columns: " + ", ".join(missing))
    data = data[REQUIRED].copy()
    for col in ["model", "transmission", "fuelType"]:
        data[col] = data[col].astype("string").str.strip()
    data = data.drop_duplicates()
    data = data[(data.price > 0) & (data.year <= 2020)].copy()
    data.loc[data.engineSize == 0, "engineSize"] = float("nan")
    return data

@st.cache_resource
def train_model(frame):
    X = frame.drop(columns="price")
    y = frame["price"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("regressor", ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    model.fit(X, y)
    return model

st.title("Ford Used Car Price Explorer")
st.caption("Explore the cleaned Ford listings and estimate a listing price (GBP) from vehicle details.")
try:
    df = load_data()
    model = train_model(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Listings", f"{len(df):,}")
c2.metric("Median listing price", f"£{df.price.median():,.0f}")
c3.metric("Price range", f"£{df.price.min():,.0f}–£{df.price.max():,.0f}")

st.subheader("Explore the data")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.price, bins=40, color="#2563eb", edgecolor="white")
    ax.set(title="Distribution of listing prices", xlabel="Price (GBP)", ylabel="Listings")
    st.pyplot(fig, clear_figure=True)
with right:
    fig, ax = plt.subplots(figsize=(7, 4))
    sample = df.sample(min(5000, len(df)), random_state=42)
    ax.scatter(sample.mileage, sample.price, alpha=0.3, s=10, color="#0f766e")
    ax.set(title="Price vs mileage", xlabel="Mileage", ylabel="Price (GBP)")
    st.pyplot(fig, clear_figure=True)

fig, ax = plt.subplots(figsize=(10, 4))
medians = df.groupby("model", observed=True).price.median().sort_values()
ax.bar(medians.index.astype(str), medians.values, color="#7c3aed")
ax.set(title="Median listing price by model", xlabel="Model", ylabel="Median price (GBP)")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

st.subheader("Predict a car's listing price")
st.write("Enter vehicle details below. The app trains an Extra Trees regression model on the project data when it starts.")
with st.form("prediction_form"):
    a, b, c = st.columns(3)
    models = sorted(df.model.dropna().astype(str).unique())
    transmissions = sorted(df.transmission.dropna().astype(str).unique())
    fuel_types = sorted(df.fuelType.dropna().astype(str).unique())
    with a:
        chosen_model = st.selectbox("Model", models)
        year = st.number_input("Year", min_value=int(df.year.min()), max_value=2020, value=2017, step=1)
        transmission = st.selectbox("Transmission", transmissions)
    with b:
        mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=30000, step=1000)
        fuel_type = st.selectbox("Fuel type", fuel_types)
        tax = st.number_input("Tax (GBP)", min_value=0, max_value=1000, value=150, step=5)
    with c:
        mpg = st.number_input("Fuel economy (mpg)", min_value=1.0, max_value=200.0, value=50.0, step=1.0)
        engine_size = st.number_input("Engine size (litres)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    submitted = st.form_submit_button("Estimate price", type="primary")

if submitted:
    features = pd.DataFrame([{
        "model": chosen_model, "year": year, "transmission": transmission,
        "mileage": mileage, "fuelType": fuel_type, "tax": tax,
        "mpg": mpg, "engineSize": engine_size,
    }])
    estimate = float(model.predict(features)[0])
    st.success(f"Estimated listing price: £{estimate:,.0f}")
    st.caption("This is an educational estimate based on historical listings; actual sale prices can differ.")
