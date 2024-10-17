import pandas as pd
import joblib
import streamlit as st

def load_data():
    try:
        df = pd.read_csv('data/LoanTapData.csv')
        if df.empty:
            st.error("The loaded dataset is empty.")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please upload the dataset.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def save_model(model):
    try:
        joblib.dump(model, 'models/loantap_model.pkl')
        st.success("Model saved successfully")
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")