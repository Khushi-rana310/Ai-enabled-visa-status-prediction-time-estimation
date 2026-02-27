from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load FULL pipeline (preprocessing + model)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        input_data = {
            "age": int(request.form["age"]),
            "work_experience_years": int(request.form["experience"]),
            "education_level": request.form["education"],
            "financial_status": request.form["financial"],
            "previous_visa_rejections": int(request.form["rejections"]),
            "document_completeness_score": float(request.form["documents"])
        }

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)