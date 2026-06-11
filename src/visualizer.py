import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x='churn', data=df, ax=ax)
    return fig

def plot_geography_analysis(df):
    fig, ax = plt.subplots()
    sns.countplot(x='geography', data=df, ax=ax)
    return fig

def plot_product_analysis(df):
    fig, ax = plt.subplots()
    sns.countplot(x='products_number', data=df, ax=ax)
    return fig
