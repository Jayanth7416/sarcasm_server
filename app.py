import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import shap
from transformers_interpret import SequenceClassificationExplainer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tensorflow.keras.models import load_model
import pickle


# Load the transformer model and tokenizer
transformer_tokenizer = AutoTokenizer.from_pretrained("./sarcasm_model")
transformer_model = AutoModelForSequenceClassification.from_pretrained("./sarcasm_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model.to(device)

# Load the LSTM model and tokenizer
lstm_model = load_model("./sarcasm_model_lstm/tf_model.h5")

with open("./sarcasm_model_lstm/tokenizer.pkl", "rb") as f:
    lstm_tokenizer = pickle.load(f)


def predict_with_transformer(text):
    """Predict with the Transformer model."""
    tokenized_input = transformer_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = transformer_model(**tokenized_input)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Sarcastic" if prediction == 1 else "Not Sarcastic"


def predict_with_lstm(text):
    """Predict with the LSTM model."""
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded_sequence = np.array(sequence)  # Adjust padding size if needed
    prediction = lstm_model.predict(padded_sequence)
    return "Sarcastic" if prediction > 0.5 else "Not Sarcastic"


def unified_prediction(text):
    """Combine predictions from both models."""
    transformer_result = predict_with_transformer(text)
    lstm_result = predict_with_lstm(text)
    return "Sarcastic" if "Sarcastic" in [transformer_result, lstm_result] else "Not Sarcastic"


# Streamlit app UI
st.title("Sarcasm Detection App")
st.write("Enter a sentence to check if it's sarcastic or not.")

# User input
text_input = st.text_area("Enter Text Here", key="text_input", height=150)
predict_button = st.button("Predict")

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "transformer_shap_values" not in st.session_state:
    st.session_state["transformer_shap_values"] = None
if "tokenized_input" not in st.session_state:
    st.session_state["tokenized_input"] = None


def update_prediction():
    if text_input:
        # Transformer Prediction with SHAP explainability
        explainer = SequenceClassificationExplainer(transformer_model, transformer_tokenizer)
        transformer_label = predict_with_transformer(text_input)
        shap_values = explainer(text_input)

        # Unified prediction logic
        unified_label = unified_prediction(text_input)

        # Update session state
        st.session_state["prediction"] = unified_label
        st.session_state["transformer_shap_values"] = shap_values
        st.session_state["tokenized_input"] = transformer_tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True, max_length=512
        )


if predict_button:
    update_prediction()

# Display unified prediction
if st.session_state["prediction"]:
    prediction_text = f"Prediction: <span style='font-size: 40px;'>"

    # Set color based on prediction
    if st.session_state["prediction"] == "Sarcastic":
        prediction_text += "<span style='color: red;'>Sarcastic</span>"
    else:
        prediction_text += "<span style='color: green;'>Not Sarcastic</span>"

    prediction_text += "</span>"
    st.markdown(prediction_text, unsafe_allow_html=True)


# Show SHAP visualization (if any)
if st.session_state["transformer_shap_values"]:
    shap_values = st.session_state["transformer_shap_values"]

    # Check if shap_values is a tuple or not, and extract the correct elements
    if isinstance(shap_values, tuple):
        shap_values = shap_values[0]  # Extract the first element if it's a tuple

    # Extract SHAP values (the numeric part of the token-value pairs)
    values = np.array([pair[1] for pair in shap_values])  # Extract only the SHAP values

    # Reshape values into a 2D matrix (1 sample, n_features)
    values_reshaped = values.reshape(1, -1)  # Reshape to (1, n_features)

    # Convert token IDs to tokens
    tokens = transformer_tokenizer.convert_ids_to_tokens(st.session_state.tokenized_input['input_ids'].squeeze().tolist())

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
    shap.summary_plot(values_reshaped, st.session_state.tokenized_input['input_ids'])







#
# # SHAP Visualization
# if st.session_state["transformer_shap_values"]:
#     shap_values = st.session_state["transformer_shap_values"]
#     tokens = transformer_tokenizer.convert_ids_to_tokens(
#         st.session_state["tokenized_input"]["input_ids"].squeeze().tolist()
#     )
#
#
#     def get_shap_color(shap_values):
#         # Ensure shap_values are numeric (float)
#         shap_values = np.array(shap_values, dtype=float)
#
#         # Normalize values for color mapping
#         max_value = max(abs(np.min(shap_values)), abs(np.max(shap_values)))
#         norm = mcolors.Normalize(vmin=-max_value, vmax=max_value)
#
#         # Use the red and green colormaps
#         colors = []
#         for val in shap_values:
#             if val > 0:
#                 color = plt.cm.Greens(norm(val))  # Green for positive values
#             else:
#                 color = plt.cm.Reds(norm(val))  # Red for negative values
#             colors.append(mcolors.rgb2hex(color[:3]))  # Convert to hex color
#
#         return colors
#
#
#     # In the part where you're generating the colored tokens
#     colored_tokens = [
#         f'<span style="color: {color}">{tok}</span>' for tok, color in zip(tokens, get_shap_color(values))
#     ]
#
#     st.markdown(" ".join(colored_tokens), unsafe_allow_html=True)