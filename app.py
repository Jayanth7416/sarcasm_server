import os
import subprocess
import sys


def install_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--user"])


# List of required packages
required_packages = [
    "streamlit",
    "transformers",
    "torch",
    "shap",
    "transformers-interpret",
    "numpy",
    "matplotlib",
    "gdown",
]

# Install all required packages
for package in required_packages:
    try:
        install_package(package)
    except Exception as e:
        print(f"Error installing {package}: {e}")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import shap
import time
import threading
from transformers_interpret import SequenceClassificationExplainer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import gdown
import zipfile



def download_and_extract_model():
    if not os.path.exists("sarcasm_model"):
        # Download the folder as a zip file
        zip_path = "sarcasm_model.zip"
        gdown.download(f"https://drive.google.com/file/d/1XYaRwxaqwRsC-VGPW6OqcfM3jP4PnvR5/view?usp=sharing", zip_path, quiet=False)
        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("sarcasm_model")
        os.remove(zip_path)  # Cleanup the zip file after extraction

# Call the function to download and set up the model
download_and_extract_model()


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./sarcasm_model")
model = AutoModelForSequenceClassification.from_pretrained("./sarcasm_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up SHAP Explainer
explainer = shap.Explainer(model, tokenizer)


def predict_and_explain(text):
    if text:
        # Initialize the explainer
        explainer = SequenceClassificationExplainer(model, tokenizer)

        # Get the model prediction
        with torch.no_grad():
            tokenized_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                device)
            outputs = model(**tokenized_input)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            label = "Sarcastic" if prediction == 1 else "Not Sarcastic"

        # Explain the prediction
        shap_values = explainer(text)  # Pass raw text directly to explainer

        return label, shap_values, tokenized_input  # Return tokenized_input for visualization

    return None, None, None


# Streamlit app UI
st.title("Sarcasm Detection App")
st.write("Enter a sentence to check if it's sarcastic or not.")

# User input field
text_input = st.text_area("Enter Text Here", key="text_input", height=150)

# Button to trigger prediction
predict_button = st.button("Predict")

# Session state to hold prediction result and visualization
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "shap_values" not in st.session_state:
    st.session_state["shap_values"] = None
if "tokenized_input" not in st.session_state:
    st.session_state["tokenized_input"] = None


# Function to call after the user clicks the predict button
def update_prediction():
    if st.session_state.text_input != "":
        label, shap_values, tokenized_input = predict_and_explain(st.session_state.text_input)
        st.session_state.prediction = label
        st.session_state.shap_values = shap_values
        st.session_state.tokenized_input = tokenized_input  # Save tokenized input for visualization


# Run prediction when the button is clicked
if predict_button:
    update_prediction()

# Display prediction with increased size and color
if st.session_state.prediction:
    prediction_text = f"Prediction: <span style='font-size: 40px;'>"

    # Set color based on the prediction
    if st.session_state.prediction == "Sarcastic":
        prediction_text += "<span style='color: red;'>Sarcastic</span>"
    else:
        prediction_text += "<span style='color: green;'>Not Sarcastic</span>"

    prediction_text += "</span>"

    # Render the prediction with the style
    st.markdown(prediction_text, unsafe_allow_html=True)

# Show SHAP visualization (if any)
if st.session_state.shap_values:
    shap_values = st.session_state.shap_values

    # Check if shap_values is a tuple or not, and extract the correct elements
    if isinstance(shap_values, tuple):
        shap_values = shap_values[0]  # Extract the first element if it's a tuple

    # Extract SHAP values (the numeric part of the token-value pairs)
    values = np.array([pair[1] for pair in shap_values])  # Extract only the SHAP values

    # Reshape values into a 2D matrix (1 sample, n_features)
    values_reshaped = values.reshape(1, -1)  # Reshape to (1, n_features)

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(st.session_state.tokenized_input['input_ids'].squeeze().tolist())

    # Function to map SHAP values to colors using gradients (light and dark shades)
    def get_shap_color(value):
        # Normalize values for color mapping
        max_value = max(abs(values.min()), abs(values.max()))
        norm = mcolors.Normalize(vmin=-max_value, vmax=max_value)

        # Use the red and green colormaps
        if value > 0:
            color = plt.cm.Greens(norm(value))  # Green for positive values
        else:
            color = plt.cm.Reds(norm(value))  # Red for negative values
        return mcolors.rgb2hex(color[:3])  # Convert to hex color

    # Create a colored HTML representation of the tokens with their SHAP value-based colors
    colored_tokens = [
        f'<span style="color: {get_shap_color(val)}">{tok}</span>' for tok, val in zip(tokens, values)
    ]

    # Display the sentence with colors based on SHAP values
    st.write("Highlighted Sentence:")
    st.markdown(" ".join(colored_tokens), unsafe_allow_html=True)

    # Debugging: Print the structure of shap_values to inspect
    st.write("SHAP Values Structure:", shap_values)

    # Debugging: Print the extracted shap_values to inspect its attributes
    st.write("Extracted SHAP Values:", shap_values)

    # Visualize SHAP summary plot using the reshaped values
    shap.summary_plot(values_reshaped, st.session_state.tokenized_input['input_ids'])  # Corrected plotting
