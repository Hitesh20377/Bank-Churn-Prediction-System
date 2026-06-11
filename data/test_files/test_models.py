from src.trainer import train_model
import pandas as pd

def test_training():
    df = pd.DataFrame({
        "Feature1": [1,2,3,4],
        "Exited": [0,1,0,1]
    })

    model, _, _ = train_model(df)
    assert model is not None