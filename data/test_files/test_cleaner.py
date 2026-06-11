from src.cleaner import clean_data
import pandas as pd

def test_clean():
    df = pd.DataFrame({
        "CustomerId": [1, 2],
        "Balance": [100, 200]
    })

    cleaned = clean_data(df)
    assert "CustomerId" not in cleaned.columns