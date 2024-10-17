import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib

from src.data_preprocessing import preprocess_data, ensure_data_types
from src.model import train_model, evaluate_model
from src.utils import load_data, save_model

@st.cache_data
def load_cached_data():
    return load_data()

def main():
    st.title("LoanTap Credit Underwriting App")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Prediction"])

    if page == "Home":
        show_home()
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "Model Training":
        show_model_training()
    elif page == "Prediction":
        show_prediction()

def show_home():
    st.write("Welcome to the LoanTap Credit Underwriting App!")
    st.write("Use the sidebar to navigate between different sections of the app.")

def show_data_analysis():
    st.header("Data Analysis")
    df = load_cached_data()
    if df.empty:
        return
    st.write("Sample data:", df.head())
    st.write("Basic statistics:", df.describe())

    st.write("Correlation Heatmap:")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("Distribution of Loan Status:")
    fig, ax = plt.subplots()
    df['loan_status'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.write("Loan Amount Distribution:")
    fig, ax = plt.subplots()
    sns.histplot(df['loan_amnt'], kde=True, ax=ax)
    st.pyplot(fig)

def show_model_training():
    st.header("Model Training")
    df = load_cached_data()
    if df.empty:
        return

    try:
        X, y = preprocess_data(df)
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = train_model(X_train, y_train)
                if model is not None:
                    save_model(model)
                    st.success("Model trained and saved successfully!")
                else:
                    st.error("Model training failed. Please check the logs for more information.")

        if st.button("Evaluate Model"):
            try:
                model = joblib.load('models/loantap_model.pkl')
                metrics = evaluate_model(model, X_test, y_test)
                st.write("Model Evaluation Metrics:")
                st.write(metrics)
            except FileNotFoundError:
                st.error("Model file not found. Please train the model first.")
            except Exception as e:
                st.error(f"Error during model evaluation: {str(e)}")
    except Exception as e:
        st.error(f"Error during data preprocessing or model training: {str(e)}")
        
def show_prediction():
    st.header("Loan Prediction")
    try:
        model = joblib.load('models/loantap_model.pkl')
        hashing_cols = joblib.load('models/hashing_cols.pkl')
        numeric_columns = joblib.load('models/numeric_columns.pkl')

        # Create input fields for all numeric columns
        input_data = {}
        for col in numeric_columns:
            input_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", value=0.0)

        # Create input fields for all hashing columns
        for col in hashing_cols:
            input_data[col] = st.selectbox(f"{col.replace('_', ' ').title()}", ['A', 'B', 'C', 'D', 'E'])

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_processed = preprocess_data(input_df, is_training=False)
            
            prediction = model.predict(input_processed)
            probability = model.predict_proba(input_processed)[0][1]

            st.write(f"Prediction: {'Approved' if prediction[0] == 1 else 'Rejected'}")
            st.write(f"Probability of approval: {probability:.2f}")
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Please check if all required model files are present and the input data is correctly formatted.")

        
if __name__ == "__main__":
    main()