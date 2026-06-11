def clean_data(df):
    # remove empty rows
    df = df.dropna(how='all')

    # clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # strip string values
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # fill missing
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown")

    print("✅ Data Cleaned")
    return df