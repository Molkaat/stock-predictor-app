import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import traceback

# Configure page
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

def load_model_safely():
    """Load the model with error handling for version compatibility issues"""
    try:
        import joblib
        model = joblib.load("rf_model.pkl")
        return model, None
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            error_msg = (
                "⚠️ NumPy version incompatibility detected. This typically happens when the model was "
                "created with a different NumPy version than what's installed on Streamlit Cloud. "
                "Please try re-training your model in an environment with matching package versions."
            )
        else:
            error_msg = f"⚠️ Error loading model: {str(e)}"
        return None, error_msg
    except Exception as e:
        return None, f"⚠️ Error loading model: {str(e)}"

if st.button("📊 Predict Next-Day Trend"):
    if input_df is not None and not input_df.empty:
        try:
            with st.spinner("Loading model and making prediction..."):
                model, error = load_model_safely()
                
                if error:
                    st.error(error)
                    st.info("💡 Solution: Try uploading a model trained with the same packages specified in requirements.txt")
                else:
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
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please provide valid input data.")

st.sidebar.markdown("---")
st.sidebar.markdown("🧠 [GitHub Repository](https://github.com/JennyWu0630/stock-predictor-app/)")
st.sidebar.markdown("👤 Author: Jianyi Wu")
st.sidebar.markdown("📧 Contact: jianyi.w@wustl.edu")
