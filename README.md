# LoanTap Credit Underwriting Project

This project implements a credit underwriting system for LoanTap, focusing on determining the creditworthiness of individuals for personal loans.

## Project Structure

- `data/`: Contains the dataset (LoanTapData.csv)
- `notebooks/`: Jupyter notebook for data analysis (LoanTap_Analysis.ipynb)
- `src/`: Source code for data preprocessing, model training, and utility functions
- `models/`: Stores the trained model
- `app.py`: Streamlit application for model deployment and interaction
- `requirements.txt`: List of required Python packages

## Setup and Installation

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Usage

The Streamlit app provides four main sections:

1. Home: Welcome page
2. Data Analysis: Displays basic statistics and visualizations of the dataset
3. Model Training: Allows users to train and evaluate the model
4. Prediction: Enables users to input loan application details and get predictions

## Model

The project uses a Logistic Regression model for credit underwriting. The model is trained on historical loan data and predicts whether a loan application should be approved or rejected.

## Data

The dataset (LoanTapData.csv) contains various features related to loan applications, including loan amount, interest rate, employment information, and credit history.

## Contributing

Feel free to fork the project and submit pull requests for any improvements or bug fixes.

## License

This project is open-source and available under the MIT License.