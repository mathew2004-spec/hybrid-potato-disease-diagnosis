import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Leaf Disease Detector",
    page_icon="🥔",
    layout="centered"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "potato_leaf_model.h5"
CLASS_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

# Confidence threshold for hybrid decision making
CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_indices():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)


model = load_model()
class_indices = load_class_indices()

# Convert class dictionary into ordered labels
class_labels = list(class_indices.keys())


# ============================================================
# EXPERT SYSTEM KNOWLEDGE BASE
# ============================================================

RULES = {

    "early_blight": {
        "name": "Early Blight",
        "symptoms": (
            "Concentric brown spots, papery or older leaves, "
            "yellow halo, and target-like spots."
        ),
        "treatment": (
            "Remove infected leaves; apply appropriate protective "
            "fungicide management such as mancozeb or copper-based "
            "products where locally approved; avoid excessive leaf wetness."
        )
    },

    "late_blight": {
        "name": "Late Blight",
        "symptoms": (
            "Water-soaked lesions, irregular dark patches, "
            "grey or brown lesions, and fuzzy growth under humid conditions."
        ),
        "treatment": (
            "Remove and destroy severely infected tissue; improve "
            "field drainage and avoid prolonged leaf wetness. "
            "Use an appropriate fungicide program such as metalaxyl- "
            "or cymoxanil-based products where locally approved."
        )
    },

    "healthy": {
        "name": "Healthy",
        "symptoms": (
            "Green, healthy leaf with no significant visible "
            "disease symptoms."
        ),
        "treatment": (
            "No disease treatment required. Continue regular monitoring "
            "and maintain normal crop-management practices."
        )
    }
}


# ============================================================
# HELPER FUNCTION
# ============================================================

def normalize_label(label):
    """
    Converts labels such as:
    Early_Blight -> early_blight
    Late Blight  -> late_blight
    Healthy      -> healthy
    """
    return label.lower().strip().replace(" ", "_")


# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # Convert image to RGB
    img = image.convert("RGB")

    # Resize to model input size
    img = img.resize(IMG_SIZE)

    # Convert to NumPy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # CNN prediction
    prediction = model.predict(img_array, verbose=0)

    # Highest probability
    predicted_index = np.argmax(prediction)

    confidence = float(prediction[0][predicted_index])

    # Predicted class
    label = class_labels[predicted_index]

    return label, confidence, prediction[0]


# ============================================================
# HYBRID EXPERT SYSTEM
# ============================================================

def expert_system(label, confidence):

    normalized_label = normalize_label(label)

    # --------------------------------------------------------
    # High confidence prediction
    # --------------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        final_classification = normalized_label
        status = "Confident"

    # --------------------------------------------------------
    # Low confidence prediction
    # --------------------------------------------------------

    else:

        final_classification = normalized_label
        status = "Needs Expert Review"

    # Get expert-system information
    rule = RULES.get(
        final_classification,
        {
            "name": label,
            "symptoms": "No rule available for this class.",
            "treatment": "Please consult an agricultural expert."
        }
    )

    return status, rule


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("🥔 Potato Leaf Disease Detector")

st.write(
    "Upload a potato leaf image to detect the disease type "
    "and receive expert-system-based recommendations."
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Choose a potato leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Potato Leaf",
        width=500
    )

    st.divider()

    # ========================================================
    # PREDICTION
    # ========================================================

    with st.spinner("Analyzing leaf image..."):

        label, confidence, probabilities = predict_image(image)

        status, rule = expert_system(
            label,
            confidence
        )

    # ========================================================
    # CLASSIFICATION RESULT
    # ========================================================

    st.subheader("🔬 Classification Result")

    # AI Prediction
    st.markdown(
        f"**AI Prediction:** `{label}`"
    )

    # Confidence
    st.markdown(
        f"**Confidence:** `{confidence:.4f}` "
        f"({confidence * 100:.2f}%)"
    )

    # Status
    st.markdown(
        f"**Status:** `{status}`"
    )

    # Final classification
    st.markdown(
        f"**Final Classification:** `{rule['name']}`"
    )

    # ========================================================
    # CONFIDENCE MESSAGE
    # ========================================================

    if confidence >= CONFIDENCE_THRESHOLD:

        st.success(
            f"High-confidence prediction: {rule['name']}"
        )

    else:

        st.warning(
            "The CNN confidence is below the predefined "
            "threshold. Expert verification is recommended."
        )

    # ========================================================
    # EXPERT SYSTEM RESULT
    # ========================================================

    st.subheader("🧠 Expert System Recommendation")

    st.markdown(
        f"**Symptoms:** {rule['symptoms']}"
    )

    st.markdown(
        f"**Treatment:** {rule['treatment']}"
    )

    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    st.subheader("📊 Model Prediction Probabilities")

    probability_data = {}

    for i, class_name in enumerate(class_labels):
        probability_data[class_name] = float(
            probabilities[i]
        )

    st.bar_chart(probability_data)

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander("ℹ️ Model Information"):

        st.write(
            "Input image size: 224 × 224 pixels"
        )

        st.write(
            "Normalization: Pixel values scaled to [0, 1]"
        )

        st.write(
            f"Confidence threshold: "
            f"{CONFIDENCE_THRESHOLD:.0%}"
        )

        st.write(
            "Classification classes:"
        )

        for class_name in class_labels:
            st.write(f"- {class_name}")