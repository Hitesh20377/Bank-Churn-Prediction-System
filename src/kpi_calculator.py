def engagement_retention_ratio(df):
    return df.shape[0]

def product_depth_index(df):
    return len(df.columns)

def high_balance_disengagement_rate(df):
    return df.select_dtypes(include='number').mean().mean()

def relationship_strength_index(df):
    return df.select_dtypes(include='number').sum().sum()