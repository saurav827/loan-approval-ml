from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Not Approved ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
