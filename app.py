from flask import Flask, render_template, request, redirect, session, Response
import pickle
import pandas as pd
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
model_accuracy = 50.0


# ================= DATABASE INIT =================
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender REAL,
            married REAL,
            income REAL,
            loan_amount REAL,
            credit_history REAL,
            result TEXT,
            confidence REAL,
            date TEXT
        )
    """)

    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("admin", generate_password_hash("1234"))
        )

    conn.commit()
    conn.close()

init_db()


# ================= ROUTES =================

@app.route("/")
def home():
    if "user" in session:
        return render_template("index.html")
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect("/admin")
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


@app.route("/admin")
def admin():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    data = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result='Loan Approved ✅'")
    approved = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result='Loan Not Approved ❌'")
    rejected = cursor.fetchone()[0]

    conn.close()

    return render_template("admin.html",
                           data=data,
                           total=total,
                           approved=approved,
                           rejected=rejected)


@app.route("/delete/<int:id>")
def delete(id):
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/admin")


@app.route("/export")
def export():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    conn.close()

    def generate():
        yield "ID,Gender,Married,Income,Loan,Result,Confidence,Date\n"
        for row in rows:
            yield ",".join(map(str, row)) + "\n"

    return Response(generate(),
                    mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=data.csv"})


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/login")

    features = [float(x) for x in request.form.values()]

    final_features = pd.DataFrame([features], columns=[
        "Gender",
        "Married",
        "ApplicantIncome",
        "LoanAmount",
        "Credit_History"
    ])

    prediction = model.predict(final_features)
    probability = model.predict_proba(final_features)
    confidence = round(max(probability[0]) * 100, 2)

    result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Not Approved ❌"

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions 
        (gender, married, income, loan_amount, credit_history, result, confidence, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        features[0],
        features[1],
        features[2],
        features[3],
        features[4],
        result,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

    return render_template("index.html",
                           prediction_text=result,
                           confidence_text=f"Confidence: {confidence}%",
                           accuracy_text=f"Model Accuracy: {model_accuracy}%")


if __name__ == "__main__":
    app.run(debug=True)