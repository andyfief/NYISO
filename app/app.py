from flask import Flask, request, jsonify
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = joblib.load("../models/xgboost_model.pkl")

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        # Parse JSON request
        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame(data)  # expecting a list of feature dicts

        # Run prediction
        prediction = model.predict(input_df)

        # Return as JSON
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)

"""
currently accepting as input,
hour             int64
dayofweek        int64
month            int64
year             int64
dayofyear        int64
dayofmonth       int64
weekofyear       int64
seasonNum        int64
Temperature    float64
averageTemp    float64
"""