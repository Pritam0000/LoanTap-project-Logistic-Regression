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

    # Check if a trained model already exists
    import os
    model_exists = os.path.exists('models/loantap_model.pkl')

    if model_exists:
        st.success("✅ A trained model already exists!")
        st.info("You can use the existing model for predictions or retrain a new model below.")
        st.warning("⚠️ If your predictions show inconsistent results (high probability but wrong decision), please retrain the model with the updated encoding.")

        # Show model info
        try:
            model = joblib.load('models/loantap_model.pkl')
            st.write(f"**Model Type:** {type(model).__name__}")
            if hasattr(model, 'get_params'):
                st.write("**Model Parameters:**")
                params = model.get_params()
                st.json({k: str(v) for k, v in params.items()})
        except Exception as e:
            st.warning(f"Could not load model details: {str(e)}")
    else:
        st.warning("⚠️ No trained model found. Please train a model first.")

    try:
        X, y = preprocess_data(df)
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train/Retrain button
        button_text = "Retrain Model" if model_exists else "Train Model"
        if st.button(button_text):
            with st.spinner("Training model..."):
                model = train_model(X_train, y_train)
                if model is not None:
                    save_model(model)
                    st.success("Model trained and saved successfully!")
                    st.balloons()
                else:
                    st.error("Model training failed. Please check the logs for more information.")

        # Evaluate button - only show if model exists
        if model_exists or os.path.exists('models/loantap_model.pkl'):
            if st.button("Evaluate Model"):
                try:
                    model = joblib.load('models/loantap_model.pkl')
                    metrics = evaluate_model(model, X_test, y_test)
                    st.write("### Model Evaluation Metrics:")

                    # Display metrics in a nice format
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col2:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                        st.metric("F1-Score", f"{metrics['f1-score']:.4f}")
                    with col3:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

                except FileNotFoundError:
                    st.error("Model file not found. Please train the model first.")
                except Exception as e:
                    st.error(f"Error during model evaluation: {str(e)}")
    except Exception as e:
        st.error(f"Error during data preprocessing or model training: {str(e)}")
        
def show_prediction():
    st.header("Loan Prediction")

    # Check if model exists
    import os
    if not os.path.exists('models/loantap_model.pkl'):
        st.error("❌ No trained model found! Please go to 'Model Training' section and train a model first.")
        return

    st.info("📋 **Fill in the loan application details below to get a prediction**")

    try:
        model = joblib.load('models/loantap_model.pkl')
        hashing_cols = joblib.load('models/hashing_cols.pkl')
        numeric_columns = joblib.load('models/numeric_columns.pkl')

        # Show required inputs
        with st.expander("ℹ️ Click here to see all required input fields"):
            st.write("**Categorical Fields:**")
            for col in hashing_cols:
                st.write(f"- {col.replace('_', ' ').title()}")
            st.write("\n**Numeric Fields:**")
            for col in numeric_columns:
                st.write(f"- {col.replace('_', ' ').title()}")

        # Field descriptions
        field_descriptions = {
            'loan_amnt': 'The total amount of the loan',
            'term': 'Loan term in months (e.g., 36 or 60)',
            'int_rate': 'Interest rate on the loan (%)',
            'installment': 'Monthly payment amount',
            'annual_inc': 'Annual income of the borrower',
            'dti': 'Debt-to-income ratio',
            'open_acc': 'Number of open credit accounts',
            'pub_rec': 'Number of derogatory public records',
            'revol_bal': 'Total credit revolving balance',
            'revol_util': 'Revolving line utilization rate (%)',
            'total_acc': 'Total number of credit lines',
            'mort_acc': 'Number of mortgage accounts',
            'pub_rec_bankruptcies': 'Number of public record bankruptcies',
            'grade': 'Loan grade (A-G, where A is best)',
            'sub_grade': 'Loan sub-grade (A1-G5)',
            'home_ownership': 'Home ownership status',
            'verification_status': 'Income verification status',
            'purpose': 'Purpose of the loan'
        }

        # Organize inputs into tabs
        tab1, tab2 = st.tabs(["📊 Loan Details", "👤 Borrower Information"])

        input_data = {}

        with tab1:
            st.subheader("Loan Information")
            col1, col2 = st.columns(2)

            with col1:
                if 'loan_amnt' in numeric_columns:
                    input_data['loan_amnt'] = st.number_input(
                        "Loan Amount ($)",
                        min_value=0.0,
                        value=10000.0,
                        step=1000.0,
                        help=field_descriptions.get('loan_amnt', '')
                    )

                if 'term' in numeric_columns:
                    term_option = st.selectbox("Loan Term (months)", [36, 60])
                    input_data['term'] = float(term_option)

                if 'int_rate' in numeric_columns:
                    input_data['int_rate'] = st.number_input(
                        "Interest Rate (%)",
                        min_value=0.0,
                        max_value=30.0,
                        value=10.0,
                        step=0.1,
                        help=field_descriptions.get('int_rate', '')
                    )

            with col2:
                if 'installment' in numeric_columns:
                    input_data['installment'] = st.number_input(
                        "Monthly Installment ($)",
                        min_value=0.0,
                        value=300.0,
                        step=10.0,
                        help=field_descriptions.get('installment', '')
                    )

                if 'purpose' in hashing_cols:
                    input_data['purpose'] = st.selectbox(
                        "Loan Purpose",
                        ['debt_consolidation', 'credit_card', 'home_improvement', 'other',
                         'major_purchase', 'small_business', 'car', 'medical', 'moving', 'vacation'],
                        help=field_descriptions.get('purpose', '')
                    )

        with tab2:
            st.subheader("Borrower Information")
            col1, col2 = st.columns(2)

            with col1:
                if 'annual_inc' in numeric_columns:
                    input_data['annual_inc'] = st.number_input(
                        "Annual Income ($)",
                        min_value=0.0,
                        value=50000.0,
                        step=5000.0,
                        help=field_descriptions.get('annual_inc', '')
                    )

                if 'dti' in numeric_columns:
                    input_data['dti'] = st.number_input(
                        "Debt-to-Income Ratio",
                        min_value=0.0,
                        max_value=100.0,
                        value=15.0,
                        step=1.0,
                        help=field_descriptions.get('dti', '')
                    )

                if 'home_ownership' in hashing_cols:
                    input_data['home_ownership'] = st.selectbox(
                        "Home Ownership",
                        ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
                        help=field_descriptions.get('home_ownership', '')
                    )

                if 'verification_status' in hashing_cols:
                    input_data['verification_status'] = st.selectbox(
                        "Income Verification Status",
                        ['Verified', 'Source Verified', 'Not Verified'],
                        help=field_descriptions.get('verification_status', '')
                    )

            with col2:
                if 'grade' in hashing_cols:
                    input_data['grade'] = st.selectbox(
                        "Loan Grade",
                        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                        help=field_descriptions.get('grade', '')
                    )

                if 'sub_grade' in hashing_cols:
                    selected_grade = input_data.get('grade', 'A')
                    sub_grades = [f"{selected_grade}{i}" for i in range(1, 6)]
                    input_data['sub_grade'] = st.selectbox(
                        "Loan Sub-Grade",
                        sub_grades,
                        help=field_descriptions.get('sub_grade', '')
                    )

                if 'open_acc' in numeric_columns:
                    input_data['open_acc'] = st.number_input(
                        "Open Credit Accounts",
                        min_value=0,
                        value=10,
                        step=1,
                        help=field_descriptions.get('open_acc', '')
                    )

                if 'total_acc' in numeric_columns:
                    input_data['total_acc'] = st.number_input(
                        "Total Credit Accounts",
                        min_value=0,
                        value=20,
                        step=1,
                        help=field_descriptions.get('total_acc', '')
                    )

        # Additional fields in expander
        with st.expander("🔧 Additional Fields (Optional)"):
            col1, col2, col3 = st.columns(3)

            with col1:
                for col in ['revol_bal', 'revol_util', 'mort_acc']:
                    if col in numeric_columns:
                        input_data[col] = st.number_input(
                            col.replace('_', ' ').title(),
                            min_value=0.0,
                            value=0.0,
                            help=field_descriptions.get(col, '')
                        )

            with col2:
                for col in ['pub_rec', 'pub_rec_bankruptcies']:
                    if col in numeric_columns:
                        input_data[col] = st.number_input(
                            col.replace('_', ' ').title(),
                            min_value=0,
                            value=0,
                            step=1,
                            help=field_descriptions.get(col, '')
                        )

            with col3:
                # Fill in any remaining numeric columns not yet covered
                covered_cols = ['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc',
                               'dti', 'open_acc', 'total_acc', 'revol_bal', 'revol_util',
                               'mort_acc', 'pub_rec', 'pub_rec_bankruptcies']
                for col in numeric_columns:
                    if col not in covered_cols and col not in input_data:
                        input_data[col] = st.number_input(
                            col.replace('_', ' ').title(),
                            value=0.0
                        )

        # Ensure all required columns are present
        for col in numeric_columns:
            if col not in input_data:
                input_data[col] = 0.0

        for col in hashing_cols:
            if col not in input_data:
                input_data[col] = 'A'

        st.markdown("---")

        # Predict button
        if st.button("🔮 Predict Loan Approval", type="primary"):
            with st.spinner("Analyzing loan application..."):
                input_df = pd.DataFrame([input_data])
                input_processed = preprocess_data(input_df, is_training=False)

                # Get prediction: 1 = Approved (Fully Paid), 0 = Rejected (Charged Off)
                prediction = model.predict(input_processed)
                # Get probability of approval (class 1)
                probability = model.predict_proba(input_processed)[0][1]

                # Display results
                st.markdown("### 📊 Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    if prediction[0] == 1:
                        st.success("✅ **LOAN APPROVED**")
                    else:
                        st.error("❌ **LOAN REJECTED**")

                with col2:
                    st.metric("Approval Probability", f"{probability:.2%}")

                # Show probability bar
                st.progress(probability)

                # Additional insights
                if probability > 0.7:
                    st.info("💡 High confidence in approval - Strong creditworthiness indicators")
                elif probability > 0.5:
                    st.warning("⚠️ Moderate confidence - Consider reviewing application details")
                else:
                    st.error("🚫 Low approval probability - High risk indicators present")

    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.error("Please train the model first in the 'Model Training' section.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Please check if all required model files are present and the input data is correctly formatted.")

        
if __name__ == "__main__":
    main()