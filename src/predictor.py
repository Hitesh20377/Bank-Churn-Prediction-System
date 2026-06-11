import joblib
import pandas as pd


def predict_churn(model_path, input_data):
    model = joblib.load(model_path)

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    return prediction, round(probability, 2)