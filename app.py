from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import os

app = Flask(__name__)

# Global variables for models and encoders
models = {}
encoders = {}
scaler = None
feature_columns = []

# Load data and train models
def train_models():
    global models, encoders, scaler, feature_columns
    
    try:
        if os.path.exists('census_income.csv'):
            df = pd.read_csv('census_income.csv')
            # Clean column names
            df.columns = df.columns.str.strip()

        else:
            print("Dataset not found!")
            return False

        # Preprocessing (Simplified based on typical notebook steps)
        # Drop duplicates and nulls if necessary
        df = df.drop_duplicates()
        df = df.dropna()

        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Keep track of original values for the UI dropdowns if needed
        # For simplicity, we just train here.
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        X = df.drop('annual_income', axis=1) # Assuming 'annual_income' becomes 0/1 after encoding
        y = df['annual_income']
        
        feature_columns = X.columns.tolist()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

        # Scale - only for specific models if needed, but RF/XGB don't strictly need it.
        # We'll skip scaling for simplicity unless required by specific models trained
        # But some like KNN/Logistic Regression benefit from it.
        # Let's scale for robustness.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test) # Not needed for training

        # Train Models
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        models['Logistic Regression'] = lr

        print("Training Decision Tree...")
        dt = DecisionTreeClassifier()
        dt.fit(X_train_scaled, y_train)
        models['Decision Tree'] = dt

        print("Training Random Forest...")
        rf = RandomForestClassifier()
        rf.fit(X_train_scaled, y_train)
        models['Random Forest'] = rf

        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train_scaled, y_train)
        models['XGBoost'] = xgb_model
        
        print("All models trained successfully.")
        return True

    except Exception as e:
        print(f"Error during training: {e}")
        return False

@app.route('/')
def home():
    if not models:
        success = train_models()
        if not success:
            return "Error: Could not load dataset or train models. check console."
            
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'Models not trained'}), 500

    try:
        data = request.json
        input_data = [] # List to hold processed features in order
        
        # We need to reconstruct the input vector based on feature_columns
        # The frontend will send raw values, we need to encode/scale them.
        
        # A safer way is to use a dataframe to match columns
        input_df = pd.DataFrame([data])
        
        # Encode categorical inputs
        for col, le in encoders.items():
            if col in input_df.columns and col != 'annual_income':
                # Handle unseen labels carefully or assume dropdowns provide valid ones
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                     # Fallback for unknown categories - usually map to a common one or error
                     # For simplest demo, assume valid input or map to 0
                     input_df[col] = 0 

        # Ensure correct column order
        input_df = input_df[feature_columns]
        
        # Scale
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df

        # Get prediction from selected model
        selected_model_name = data.get('model_type', 'Random Forest') # Default
        model = models.get(selected_model_name)
        
        if not model:
            return jsonify({'error': f'Model {selected_model_name} not found'}), 400

        prediction = model.predict(input_scaled)[0]
        # Decode prediction if target was categorical (it was!)
        result_label = encoders['annual_income'].inverse_transform([prediction])[0]

        return jsonify({'prediction': result_label})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    train_models()
    app.run(debug=True, port=5001)
