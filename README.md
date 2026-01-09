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

1. **Home**: Welcome page with project overview
2. **Data Analysis**: Displays basic statistics and visualizations of the dataset
   - Sample data preview
   - Correlation heatmap
   - Loan status distribution
   - Loan amount distribution

3. **Model Training**: Smart model management with existing model detection
   - ✅ **Automatically detects if a trained model exists**
   - Displays current model information and parameters
   - Option to use existing model or retrain a new one
   - Interactive test size selection
   - Model evaluation with detailed metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

4. **Prediction**: User-friendly loan approval prediction interface
   - **Organized input tabs**: Loan Details and Borrower Information
   - **Clear field descriptions** with tooltips for each input
   - **Smart sub-grade selection** based on selected grade
   - **Required fields indicator** - expandable section showing all inputs needed
   - **Visual prediction results** with approval probability and confidence levels
   - **Interactive feedback** based on prediction confidence

## Key Features & Improvements

### 🔄 Smart Model Management
- **No forced retraining**: The app detects if a trained model exists and allows you to use it
- **Model information display**: View current model parameters and type
- **Retrain option**: Easily retrain a new model if needed
- All models are saved and can be reused across sessions

### 📊 Enhanced User Interface
- **Organized input sections**: Tabbed interface separating loan and borrower information
- **Field descriptions**: Every input has a helpful tooltip explaining what it means
- **Smart validation**: Appropriate input types and ranges for each field
- **Visual feedback**: Color-coded results with confidence indicators

### 🎯 Intelligent Predictions
- **Clear results**: Shows approval/rejection with probability percentage
- **Confidence levels**: Visual indicators (High/Moderate/Low confidence)
- **Progress bar**: Visual representation of approval probability
- **Expandable field list**: See all required inputs at a glance

## Model

The project uses a Logistic Regression model for credit underwriting. The model is trained on historical loan data and predicts whether a loan application should be approved or rejected.

**Model Performance**:
- Uses RandomizedSearchCV for hyperparameter tuning
- Optimized for ROC-AUC score
- Includes feature hashing for categorical variables
- Standard scaling for numeric features

## Data

The dataset (LoanTapData.csv) contains various features related to loan applications, including loan amount, interest rate, employment information, and credit history.

## Contributing

Feel free to fork the project and submit pull requests for any improvements or bug fixes.

## License

This project is open-source and available under the MIT License.