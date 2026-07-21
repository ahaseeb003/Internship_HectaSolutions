import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

# Set page config
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/insurance.csv')
    return df

df = load_data()

# Preprocessing and model training (for prediction option)
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
X_train_processed = preprocessor.fit_transform(X_train)
model.fit(X_train_processed, y_train)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Project Overview", "Interactive Visualization", "Real-world Prediction"])

if option == "Project Overview":
    st.title("Medical Insurance Cost Analysis & Prediction")
    st.markdown("""
    This application is part of the **Regression Modeling: From Scratch to Production** internship task.
    It demonstrates the application of linear regression to predict medical insurance costs based on individual attributes.
    
    ### Key Features:
    - **Interactive Visualization**: Explore how different factors like age, BMI, and smoking status impact insurance charges.
    - **Real-world Prediction**: Use a trained linear regression model to estimate insurance costs for new individuals.
    - **Model Diagnostics**: Visual evidence of model performance and residual analysis.
    """)
    st.image("https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&q=80&w=1000", caption="Medical Insurance Analysis")

elif option == "Interactive Visualization":
    st.title("Interactive Data Visualization")
    st.markdown("Explore the relationship between features and insurance charges.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Charges")
        fig, ax = plt.subplots()
        sns.histplot(df['charges'], kde=True, ax=ax, color='skyblue')
        ax.set_title("Distribution of Medical Insurance Charges")
        st.pyplot(fig)

    with col2:
        st.subheader("Charges by Smoking Status")
        fig, ax = plt.subplots()
        sns.boxplot(x='smoker', y='charges', data=df, ax=ax, palette='Set2')
        ax.set_title("Insurance Charges: Smoker vs. Non-Smoker")
        st.pyplot(fig)

    st.divider()
    st.subheader("Dynamic Scatter Plot with Sliders")
    
    # Sliders for dynamic visualization
    age_range = st.slider("Select Age Range", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
    bmi_range = st.slider("Select BMI Range", float(df['bmi'].min()), float(df['bmi'].max()), (float(df['bmi'].min()), float(df['bmi'].max())))
    
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) & 
                     (df['bmi'] >= bmi_range[0]) & (df['bmi'] <= bmi_range[1])]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=filtered_df, ax=ax, alpha=0.7)
    ax.set_title(f"Age vs. Charges (Filtered by Age and BMI)")
    st.pyplot(fig)

elif option == "Real-world Prediction":
    st.title("Insurance Cost Prediction Tool")
    st.markdown("Enter the details below to estimate the annual medical insurance cost.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Sex", ["male", "female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        
        with col2:
            children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
            smoker = st.selectbox("Smoker", ["yes", "no"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
        
        submit_button = st.form_submit_button("Predict Cost")

    if submit_button:
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        
        st.success(f"Estimated Annual Insurance Cost: **${prediction:,.2f}**")
        
        # Breakdown visualization
        st.info("Note: This is an estimate based on a linear regression model trained on historical data.")
        
        # Comparison with average
        avg_cost = df['charges'].mean()
        diff = prediction - avg_cost
        color = "red" if diff > 0 else "green"
        st.markdown(f"Your estimated cost is <span style='color:{color}'>**${abs(diff):,.2f} {'above' if diff > 0 else 'below'}**</span> the average cost of **${avg_cost:,.2f}**.", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("Built for ML Internship Task 05")
