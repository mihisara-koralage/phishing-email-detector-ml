import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("phishing_model_new.pkl")

# ----------------------------
# Model Metrics
# ----------------------------
ACCURACY = 98.86
PRECISION = 98.44
RECALL = 99.14
F1_SCORE = 98.79

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("🛡️ AI-Powered Phishing Email Detector")

st.markdown("""
This application uses a **Linear Support Vector Machine (Linear SVM)** model
trained on the Enron phishing email dataset to classify emails as:

- ✅ Legitimate Email
- 🚨 Phishing Email
""")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Model Information")

st.sidebar.write("**Algorithm:** Linear SVM")
st.sidebar.write("**Feature Extraction:** TF-IDF")
st.sidebar.write("**Dataset:** Enron Email Dataset")
st.sidebar.write("**Total Emails:** 29,767")

# ----------------------------
# Metrics Row
# ----------------------------
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{ACCURACY}%")
col2.metric("Precision", f"{PRECISION}%")
col3.metric("Recall", f"{RECALL}%")
col4.metric("F1 Score", f"{F1_SCORE}%")

# ----------------------------
# Performance Chart
# ----------------------------
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [ACCURACY, PRECISION, RECALL, F1_SCORE]
})

fig = px.bar(
    metrics_df,
    x="Metric",
    y="Score",
    text="Score",
    title="Model Evaluation Metrics"
)

fig.update_layout(yaxis_range=[90, 100])

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Model Comparison Table
# ----------------------------
st.subheader("Model Comparison Results")

st.write(
    "Eight machine learning algorithms were evaluated using Accuracy, Precision, Recall, and F1 Score. "
    "Linear SVM achieved the highest performance and was selected for deployment."
)

comparison = pd.read_csv("model_comparison.csv")

st.expander("View Model Comparison Results").dataframe(
    comparison,
    use_container_width=True
)

# ----------------------------
# Email Input
# ----------------------------
st.subheader("Email Analysis")

email_text = st.text_area(
    "Paste an email or message below",
    height=250
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:

        prediction = model.predict([email_text])[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 PHISHING EMAIL DETECTED")

            st.markdown("""
            ### Warning Signs
            - Requests sensitive information
            - May contain suspicious links
            - Attempts to create urgency
            - Potential impersonation attempt
            """)

        else:
            st.success("✅ LEGITIMATE EMAIL")

            st.markdown("""
            ### Analysis Result
            The email appears to be legitimate based on
            patterns learned from the training dataset.
            """)

# ----------------------------
# Project Information
# ----------------------------
st.divider()

with st.expander("Project Details"):

    st.markdown("""
    ### Machine Learning Pipeline

    1. Data Collection
    2. Data Preprocessing
    3. TF-IDF Feature Extraction
    4. Model Training
    5. Model Comparison
    6. Model Selection
    7. Streamlit Deployment

    ### Compared Models

    - Linear SVM
    - Logistic Regression
    - Naive Bayes
    - Random Forest
    - Decision Tree
    - KNN
    - Gradient Boosting
    - AdaBoost

    ### Selected Model

    Linear SVM achieved the highest overall performance and was selected for deployment.
    """)