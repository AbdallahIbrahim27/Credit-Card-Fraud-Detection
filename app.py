import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    with open('lr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Main title
st.title("🔍 Fraud Detection System")
st.markdown("---")

# Load the model
try:
    model, scaler = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Create input fields
st.subheader("Enter Transaction Details")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    time = st.number_input("Time of Transaction", min_value=0, value=0)

with col2:
    v1 = st.number_input("V1 (Principal Component 1)", value=0.0)
    v2 = st.number_input("V2 (Principal Component 2)", value=0.0)

# Prediction button
if st.button("Predict Fraud"):
    try:
        # Prepare input data with zeros for unused features
        input_data = np.zeros(30)  # Initialize with zeros for all 30 features
        input_data[0] = amount    # Amount
        input_data[1] = time      # Time
        input_data[2] = v1        # V1
        input_data[3] = v2        # V2
        
        # Scale each feature individually
        scaled_data = np.zeros_like(input_data)
        for i in range(len(input_data)):
            scaled_data[i] = scaler.transform([[input_data[i]]])[0][0]
        
        # Reshape for prediction
        scaled_data = scaled_data.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error("⚠️ This transaction is likely fraudulent!")
        else:
            st.success("✅ This transaction appears to be legitimate!")
            
        st.write(f"Confidence: {probability[0][1]*100:.2f}%")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add some information about the model
st.markdown("---")
st.subheader("About")
st.write("""
This application uses machine learning to detect potentially fraudulent transactions.
The model analyzes four key features:
- Transaction Amount: The monetary value of the transaction
- Time of Transaction: When the transaction occurred
- V1: First principal component of transaction features
- V2: Second principal component of transaction features

These features are used to predict whether a transaction is likely to be fraudulent.
""")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 