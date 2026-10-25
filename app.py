
from flask import Flask, render_template, request   
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import pickle
app = Flask(__name__)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


print("Model successfully exported to 'logistic_regression_model.pkl'")
print("Scaler successfully exported to 'scaler.pkl'")
# --------------------------------------------------
# 1. Load the trained model and scaler
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


# --------------------------------------------------
# 2. Define the expected feature order
# Must match the training dataset exactly
# --------------------------------------------------

feature_cols = [
    "chol",
    "stab.glu",
    "hdl",
    "ratio",
    "age",
    "height",
    "weight",
    "bp.1s",
    "bp.1d",
    "bp.2s",
    "bp.2d",
    "waist",
    "hip",
    "time.ppn",
    "location_Louisa",
    "gender_male",
    "frame_medium",
    "frame_small",
]

numeric_fields = [
    "chol",
    "stab_glu",
    "hdl",
    "ratio",
    "age",
    "height",
    "weight",
    "bp_1s",
    "bp_1d",
    "bp_2s",
    "bp_2d",
    "waist",
    "hip",
    "time_ppn",
]


# --------------------------------------------------
# 3. Home page
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        probability=None,
        error=None,
        form_data={}
    )


# --------------------------------------------------
# 4. Predict diabetes
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect and validate numeric inputs
        data = {}

        for field in numeric_fields:
            raw_value = request.form.get(field, "").strip()

            if not raw_value:
                raise ValueError(
                    f"Please enter a value for {field.replace('_', ' ')}."
                )

            value = float(raw_value)

            if not np.isfinite(value):
                raise ValueError(
                    f"Please enter a valid number for "
                    f"{field.replace('_', ' ')}."
                )

            if value < 0:
                raise ValueError(
                    f"{field.replace('_', ' ').capitalize()} "
                    "cannot be negative."
                )

            data[field] = value

        # Map form names to the model's feature names
        input_data = {
            "chol": data["chol"],
            "stab.glu": data["stab_glu"],
            "hdl": data["hdl"],
            "ratio": data["ratio"],
            "age": data["age"],
            "height": data["height"],
            "weight": data["weight"],
            "bp.1s": data["bp_1s"],
            "bp.1d": data["bp_1d"],
            "bp.2s": data["bp_2s"],
            "bp.2d": data["bp_2d"],
            "waist": data["waist"],
            "hip": data["hip"],
            "time.ppn": data["time_ppn"],
        }

        # Validate categorical inputs
        binary_fields = [
            "location_Louisa",
            "gender_male",
            "frame_medium",
            "frame_small",
        ]

        for field in binary_fields:
            raw_value = request.form.get(field, "0").strip()

            if raw_value not in ("0", "1"):
                raise ValueError(f"Invalid selection for {field}.")

            input_data[field] = int(raw_value)

        # A frame cannot be both medium and small
        if (
            input_data["frame_medium"] == 1
            and input_data["frame_small"] == 1
        ):
            raise ValueError("Please select only one body frame size.")

        # Preserve the submitted form values if validation fails
        form_data = request.form.to_dict()

        # Create a DataFrame with exactly the required columns
        input_df = pd.DataFrame(
            [input_data],
            columns=feature_cols
        )

        # Check for missing or invalid values
        if input_df.isnull().any().any():
            raise ValueError("Some input values are missing.")

        if not np.isfinite(input_df.to_numpy(dtype=float)).all():
            raise ValueError("Please check your input values.")

        # Scale features using the original fitted scaler
        scaled_inputs = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_inputs)[0]

        # Obtain the probability of the positive class (class 1)
        if not hasattr(model, "predict_proba"):
            raise RuntimeError(
                "The loaded model does not support predict_proba()."
            )

        probabilities = model.predict_proba(scaled_inputs)[0]

        # Find the probability associated with class 1
        classes = list(model.classes_)

        if 1 not in classes:
            raise RuntimeError(
                "The model does not contain the expected positive class 1."
            )

        positive_index = classes.index(1)
        probability = float(probabilities[positive_index]) * 100

        # Format the prediction
        result_text = (
            "Diabetic" if prediction == 1 else "Non-Diabetic"
        )

        return render_template(
            "index.html",
            prediction=result_text,
            probability=f"{probability:.2f}%",
            error=None,
            form_data=form_data
        )

    except (ValueError, TypeError) as exc:
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            error=str(exc),
            form_data=request.form.to_dict()
        ), 400

    except Exception as exc:
        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            error=(
                "Prediction failed. Please verify that the model, "
                "scaler, and input features match the training setup."
            ),
            form_data=request.form.to_dict()
        ), 500


# --------------------------------------------------
# 5. Start Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
