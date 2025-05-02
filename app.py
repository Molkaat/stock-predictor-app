import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Trend Predictor", layout="centered")

st.title("📈 Next-Day Stock Trend Predictor")
st.markdown("Predict whether the stock trend tomorrow will be **Bullish**, **Bearish**, or **Stable** using a machine learning model trained on S&P 500 data.")

st.sidebar.header("📥 Input Data")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    close_price = st.number_input("Today's closing price:", min_value=0.0, format="%.2f")
    volume = st.number_input("Today's volume:", min_value=0.0)
    volatility = st.slider("Volatility (0–1):", 0.0, 1.0, step=0.01)
    lag1_return = st.number_input("Lag 1 Return (%):", step=0.01)

    input_df = pd.DataFrame([{
        "close": close_price,
        "volume": volume,
        "volatility": volatility,
        "lag1_return": lag1_return
    }])
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = None

if st.button("📊 Predict Next-Day Trend"):
    if input_df is not None and not input_df.empty:
        model = joblib.load("rf_model.pkl")
        prediction = model.predict(input_df)
        st.success(f"📉 Predicted Trend: **{prediction[0]}**")

        st.subheader("🔍 Feature Importances")
        importances = model.feature_importances_
        features = input_df.columns
        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please provide valid input data.")

st.sidebar.markdown("---")
st.sidebar.markdown("🧠 [GitHub Repository](https://github.com/JennyWu0630/stock-predictor-app/)")
st.sidebar.markdown("👤 Author: Jianyi Wu")
st.sidebar.markdown("📧 Contact: jianyi.w@wustl.edu")
