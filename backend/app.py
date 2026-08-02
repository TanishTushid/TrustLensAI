from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))


def add_profile_features(df):
    df = df.copy()

    df['followers_to_follows'] = df['#followers'] / df['#follows'].replace(0, 1)
    df['posts_per_follower'] = df['#posts'] / df['#followers'].replace(0, 1)

    df['log_description_length'] = np.log1p(df['description length'])
    df['log_posts'] = np.log1p(df['#posts'])
    df['log_followers'] = np.log1p(df['#followers'])
    df['log_follows'] = np.log1p(df['#follows'])

    return df


# Training feature list
feature_columns = [
    'profile pic',
    'nums/length username',
    'fullname words',
    'nums/length fullname',
    'name==username',
    'description length',
    'external URL',
    'private',
    '#posts',
    '#followers',
    '#follows',
    'followers_to_follows',
    'posts_per_follower',
    'log_description_length',
    'log_posts',
    'log_followers',
    'log_follows'
]

categorical_cols = [
    'profile pic',
    'name==username',
    'external URL',
    'private'
]

@app.route('/')
def home():
    return 'welcome to TrustLens'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print('request received')
        data = request.get_json()
        if not data:
            return jsonify({'error':'No JSON data received'})
        print('JSON received')
        df = pd.DataFrame([data])

        print("3. DataFrame created")

        # Feature Engineering
        df = add_profile_features(df)

        print("4. Features engineered")

        # Encode categorical features
        for col in categorical_cols:
            df[col] = df[col].astype(int)

        print("5. Encoding complete")

        # Scale
        X_scaled = pd.DataFrame(
            scaler.transform(df[feature_columns]),
            columns=feature_columns
        )

        print("6. Scaling complete")

        # Select important features
        X = X_scaled[selected_features]

        print("7. Starting prediction")

        probability = float(model.predict(X, verbose=0).flatten()[0])

        print("8. Prediction complete")

        prediction = "Fake" if probability > 0.5 else "Genuine"

        return jsonify({
            "prediction": prediction,
            "confidence": round(probability, 4)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "error": str(e)
        }), 400

@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({'error':str(e)}), 500

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000)