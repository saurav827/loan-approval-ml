from flask import Flask, render_template, request, redirect, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

model = pickle.load(open("model.pkl", "rb"))
model_accuracy = 50.0


@app.route("/")
def home():
    if "user" in session:
        return render_template("index.html")
    else:
        return redirect("/login")


# 🔹 STEP 3 — Yaha Add Karo (Login Route)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect("/")
        else:
            return "Invalid Credentials"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")



@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/login")

    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])

    prediction = model.predict(final_features)
    probability = model.predict_proba(final_features)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Not Approved ❌"

    return render_template(
        "index.html",
        prediction_text=result,
        confidence_text=f"Confidence: {confidence}%",
        accuracy_text=f"Model Accuracy: {model_accuracy}%"
    )


if __name__ == "__main__":
    app.run(debug=True)
