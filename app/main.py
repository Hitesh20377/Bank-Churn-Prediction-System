import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_raw_data
from src.trainer import train_model
from src.predictor import predict_churn


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

raw_path = os.path.join(BASE_DIR, "data", "raw", "bank_customers.csv")
processed_path = os.path.join(BASE_DIR, "data", "processed", "cleaned.csv")
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")


st.set_page_config(page_title="Bank Churn Analysis", layout="wide")

st.title("🏦 Bank Customer Retention Dashboard")


@st.cache_data
def load_data():
    df = load_raw_data(raw_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Young", "Adult", "Middle Age", "Senior"]
    )

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    return df


df = load_data()


# ---------------- KPI ----------------

total_customers = len(df)
churned_customers = int(df["churn"].sum())
retained_customers = total_customers - churned_customers
churn_rate = df["churn"].mean() * 100
retention_rate = 100 - churn_rate
active_rate = df["active_member"].mean() * 100
avg_balance = df["balance"].mean()
avg_credit_score = df["credit_score"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned_customers)
col3.metric("Retained Customers", retained_customers)
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

col5, col6, col7, col8 = st.columns(4)

col5.metric("Retention Rate", f"{retention_rate:.2f}%")
col6.metric("Active Members", f"{active_rate:.2f}%")
col7.metric("Average Balance", f"{avg_balance:.2f}")
col8.metric("Average Credit Score", f"{avg_credit_score:.2f}")


# ---------------- DATA ----------------

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(20), width="stretch")


# ---------------- DASHBOARD ----------------

st.subheader("📊 Dashboard")

c1, c2 = st.columns(2)

with c1:
    fig1 = px.histogram(
        df,
        x="country",
        color="churn",
        barmode="group",
        title="Churn by Country"
    )
    st.plotly_chart(fig1, width="stretch")

with c2:
    fig2 = px.histogram(
        df,
        x="gender",
        color="churn",
        barmode="group",
        title="Churn by Gender"
    )
    st.plotly_chart(fig2, width="stretch")


c3, c4 = st.columns(2)

with c3:
    fig3 = px.histogram(
        df,
        x="age_group",
        color="churn",
        barmode="group",
        title="Churn by Age Group"
    )
    st.plotly_chart(fig3, width="stretch")

with c4:
    fig4 = px.histogram(
        df,
        x="products_number",
        color="churn",
        barmode="group",
        title="Churn by Products Number"
    )
    st.plotly_chart(fig4, width="stretch")


c5, c6 = st.columns(2)

with c5:
    fig5 = px.box(
        df,
        x="churn",
        y="balance",
        title="Balance vs Churn"
    )
    st.plotly_chart(fig5, width="stretch")

with c6:
    fig6 = px.box(
        df,
        x="churn",
        y="credit_score",
        title="Credit Score vs Churn"
    )
    st.plotly_chart(fig6, width="stretch")


c7, c8 = st.columns(2)

with c7:
    fig7 = px.histogram(
        df,
        x="active_member",
        color="churn",
        barmode="group",
        title="Active Member vs Churn"
    )
    st.plotly_chart(fig7, width="stretch")

with c8:
    fig8 = px.histogram(
        df,
        x="credit_card",
        color="churn",
        barmode="group",
        title="Credit Card vs Churn"
    )
    st.plotly_chart(fig8, width="stretch")


# ---------------- FILTER ----------------

st.subheader("🔍 Filter Data")

country_filter = st.multiselect(
    "Select Country",
    options=df["country"].unique(),
    default=df["country"].unique()
)

gender_filter = st.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

filtered_df = df[
    (df["country"].isin(country_filter)) &
    (df["gender"].isin(gender_filter))
]

st.dataframe(filtered_df, width="stretch")


# ---------------- ML MODEL ----------------

if st.button("Train Churn Prediction Model"):

    metrics, feature_importance = train_model(df, model_path)

    st.success("Model trained successfully!")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{metrics['Accuracy']}%")
    m2.metric("Precision", f"{metrics['Precision']}%")
    m3.metric("Recall", f"{metrics['Recall']}%")
    m4.metric("F1 Score", f"{metrics['F1 Score']}%")

    st.subheader("📌 Confusion Matrix")

    cm_df = pd.DataFrame(
        metrics["Confusion Matrix"],
        index=["Actual Not Churn", "Actual Churn"],
        columns=["Predicted Not Churn", "Predicted Churn"]
    )

    st.dataframe(cm_df)

    st.subheader("🔥 Top Feature Importance")

    fig_importance = px.bar(
        feature_importance.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Churn Drivers"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

# ---------------- PREDICTION ----------------

st.subheader("🔮 Predict Customer Churn")

credit_score = st.number_input("Credit Score", 300, 900, 650)
country = st.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products_number = st.number_input("Products Number", 1, 4, 1)
credit_card = st.selectbox("Credit Card", [1, 0])
active_member = st.selectbox("Active Member", [1, 0])
estimated_salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0)

if st.button("Predict Churn Risk"):
    if not os.path.exists(model_path):
        st.warning("Train model first .. !")
    else:
        input_data = {
            "credit_score": credit_score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": credit_card,
            "active_member": active_member,
            "estimated_salary": estimated_salary
        }

        prediction, probability = predict_churn(model_path, input_data)

        if prediction == 1:
            st.error(f"High Churn Risk: {probability}%")
        else:
            st.success(f"Low Churn Risk: {probability}%")


# ---------------- INSIGHTS ----------------

st.subheader("💡 Business Insights")

st.write("""
- Germany customers show higher churn risk compared to other countries.
- Inactive members are more likely to churn.
- Customers with very high balance but low engagement need retention focus.
- Product usage and activity status are important churn indicators.
""")