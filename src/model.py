from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import streamlit as st

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import joblib
import streamlit as st

def train_model(X_train, y_train):
    try:
        # Define the parameter space
        param_distributions = {
            'C': uniform(loc=0, scale=4),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': randint(100, 1000)
        }
        
        # Create a base model
        model = LogisticRegression(random_state=42)
        
        # Randomized Search with cross-validation
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=10,  # Number of parameter settings that are sampled
            cv=3,       # Number of folds in cross-validation
            scoring='roc_auc',
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
        
        with st.spinner('Training model... This may take a few minutes.'):
            random_search.fit(X_train, y_train)
        
        st.success('Model training completed!')
        st.write(f"Best parameters: {random_search.best_params_}")
        st.write(f"Best ROC AUC score: {random_search.best_score_:.4f}")
        
        # Save the best model
        joblib.dump(random_search.best_estimator_, 'models/loantap_model.pkl')
        return random_search.best_estimator_
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None
    
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
            'roc_auc': auc_score
        }
        return metrics
    except Exception as e:
        st.error(f"Error during model evaluation: {str(e)}")
        return None