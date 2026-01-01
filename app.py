from flask import Flask, request, render_template
import pandas as pd
import joblib

# Define the custom function again
def df_lowercase(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()
    return df
app = Flask(__name__)

# Load trained pipeline + expected schema
pipeline = joblib.load("model.pkl")
expected_cols = joblib.load("columns.pkl")

# Define numeric columns (for type casting)
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {}

    # Collect values from form
    for col in expected_cols:
        val = request.form.get(col)

        if val is None:
            return f"Error: Missing input for {col}"

        # Convert numerics safely
        if col in NUMERIC_COLS:
            try:
                data[col] = float(val)
            except ValueError:
                return f"Error: Invalid numeric value for {col}"
        else:
            data[col] = val

    # Create DataFrame and align schema
    df = pd.DataFrame([data])
    df = df.reindex(columns=expected_cols)

    # Predict
    pred = pipeline.predict(df)[0]
    result = (
    "Congrats! No Heart Disease 🎉"
    if pred == 0
    else "⚠️ Some Chances Of Heart Disease "
         "(Note: this is only a prediction on your data. "
         "If you want to know actually you have heart disease "
         "or which type you have, please contact a doctor.)"
)

    return render_template("index.html", result=result, input=data)

if __name__ == "__main__":
    app.run(debug=True)

