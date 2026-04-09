from flask import Flask, request, jsonify
import joblib, json, os
from model.feature_extractor import extract_features
from utils.explain import generate_explanation
from utils.shap_explainer import get_shap_values
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)

model = joblib.load("model/phishing_model.pkl")
HISTORY_FILE = "data/history.json"

def save_history(entry):
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)

    data.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)
@app.route("/test")
def test():
    url = "http://example.com"

    features = extract_features(url)
    prediction = model.predict([features])[0]
    prob = model.predict_proba([features])[0][1]

    return {
        "url": url,
        "prediction": "Phishing" if prediction == 1 else "Safe",
        "risk_score": round(prob * 100, 2)
    }
@app.route("/predict", methods=["POST"])
def predict():
    url = request.json.get("url")

    features = extract_features(url)
    prediction = model.predict([features])[0]
    prob = model.predict_proba([features])[0][1]

    risk_score = round(prob * 100, 2)
    explanation = generate_explanation(features)
    shap_values = get_shap_values(features, model)

    result = {
        "url": url,
        "prediction": "Phishing" if prediction == 1 else "Safe",
        "risk_score": risk_score,
        "explanation": explanation,
        "shap": shap_values
    }

    save_history(result)

    return jsonify(result)

@app.route("/history", methods=["GET"])
def history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])

    with open(HISTORY_FILE, "r") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))