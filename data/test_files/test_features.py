from src.feature_engineering import engineer_features
import pandas as pd

def test_features():
    df = pd.DataFrame({
        "Geography": ["France", "Germany"],
        "Gender": ["Male", "Female"]
    })

    df_new = engineer_features(df)
    assert df_new.shape[1] > df.shape[1]