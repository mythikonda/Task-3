from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Fairness ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([
        data["age"],
        data["gender"],
        data["income"],
        data["education"]
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    return jsonify({"loan_status": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)