def create_features(df):
    # Example feature (safe check)
    if 'age' in df.columns:
        df['age_group'] = df['age'].apply(
            lambda x: 'young' if x < 30 else 'adult' if x < 60 else 'senior'
        )

    return df