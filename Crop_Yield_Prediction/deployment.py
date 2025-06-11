import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from datetime import datetime
import logging
import os

# logging
logging.basicConfig(filename='model_deployment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


df = pd.read_csv("Engineered_CropYield_Environmental_yearly.csv")
train_mask = df['Year'] <= 2019
X = df.drop(columns=['Year', 'Yield', 'Decade'])
y = df['Yield']
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]


rf_model_initial = RandomForestRegressor(max_depth=10, min_samples_split=5, n_estimators=300, random_state=42)
rf_model_initial.fit(X_train, y_train)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model_initial.feature_importances_})
top_features = feature_importance.sort_values('Importance', ascending=False).head(10)['Feature'].values
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]


rf_model = RandomForestRegressor(max_depth=12, min_samples_split=5, n_estimators=350, random_state=42)
xgb_model = XGBRegressor(learning_rate=0.3, max_depth=10, n_estimators=80, random_state=42)
rf_model.fit(X_train_selected, y_train)
xgb_model.fit(X_train_selected, y_train)


def predict_ensemble(input_data):
    rf_pred = rf_model.predict(input_data)
    xgb_pred = xgb_model.predict(input_data)
    return 0.6 * rf_pred + 0.4 * xgb_pred

# Step 1: Model Serialization
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf_model, os.path.join(model_dir, 'random_forest_model.pkl'))
joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost_model.pkl'))
logging.info("Models serialized successfully.")

# Step 2: Model Serving with Flask
app = Flask(__name__)
auth = HTTPBasicAuth()

# user
users = {
    "admin": "password123"
}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username

# API endpoint
@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "No features provided"}), 400
        
        input_df = pd.DataFrame([data['features']], columns=top_features)
        prediction = predict_ensemble(input_df)
        
        logging.info(f"Prediction request received for features: {data['features']}, Result: {prediction[0]}")
        return jsonify({"prediction": prediction[0], "unit": "kg/ha"})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Step 3: Monitoring and Logging
test_pred = predict_ensemble(X_test_selected)
mae = mean_absolute_error(y_test, test_pred)
mse = mean_squared_error(y_test, test_pred)
r2 = r2_score(y_test, test_pred)
logging.info(f"Model performance on test set - MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)