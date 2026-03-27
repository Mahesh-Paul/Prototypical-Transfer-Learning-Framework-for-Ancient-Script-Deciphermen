import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path

st.set_page_config(page_title="Indus Valley Script Recognizer", layout="wide")

st.title("🏺 Indus Valley Script Recognition")
st.markdown("### Few-Shot Learning with Prototypical Networks")
st.markdown("Trained on Greek → Applied to Indus Valley Script")

# ============================================
# SETUP PATHS
# ============================================

DATASETS_PATH = Path(r"C:\Users\mp672\OneDrive\Desktop\Few_shot_learning\Git_file\few-shot-ivc-master\few-shot-ivc-master\datasets")
TEST_PATH = DATASETS_PATH / "MahadevanData_Raw" / "images_evaluation"

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    class CNNEncoder(nn.Module):
        def __init__(self):
            super(CNNEncoder, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)
            x = torch.mean(x, dim=(2, 3))
            return x
    
    class PrototypicalNetwork(nn.Module):
        def __init__(self, encoder):
            super(PrototypicalNetwork, self).__init__()
            self.encoder = encoder
        
        def forward(self, support_images, support_labels, query_images):
            support_features = self.encoder(support_images)
            query_features = self.encoder(query_images)
            
            prototypes = []
            unique_labels = torch.unique(support_labels)
            for label in unique_labels:
                prototype = support_features[support_labels == label].mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)
            
            distances = torch.cdist(query_features, prototypes)
            return -distances
    
    encoder = CNNEncoder()
    model = PrototypicalNetwork(encoder)
    
    # Load trained weights
    model_path = Path("indus_valley_few_shot_model.pth")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Trained model not found. Using untrained model for demo.")
        st.info("Please save your trained model as 'indus_valley_few_shot_model.pth'")
    
    model.eval()
    return model

# ============================================
# LOAD SUPPORT EXAMPLES (NO CACHING - FRESH EACH TIME)
# ============================================

def load_support_examples(n_shot=5):
    """
    Load real support examples from test dataset
    This runs fresh every time - no caching
    """
    # Check if test path exists
    if not TEST_PATH.exists():
        st.error(f"Test path not found: {TEST_PATH}")
        return None, None, None, None
    
    # Get all symbol folders
    symbol_folders = [f for f in TEST_PATH.iterdir() if f.is_dir()]
    
    if len(symbol_folders) == 0:
        st.error("No symbol folders found in test path")
        return None, None, None, None
    
    # Randomly select n_shot folders (classes)
    selected_folders = random.sample(symbol_folders, min(n_shot, len(symbol_folders)))
    
    support_images = []
    support_labels = []
    support_display = []
    class_names = []
    
    for label_idx, folder in enumerate(selected_folders):
        # Get all images in this folder
        images = list(folder.glob("*.png")) + list(folder.glob("*.PNG"))
        
        if images:
            img_path = images[0]
            class_names.append(folder.name)
            
            # Load and preprocess image
            img = Image.open(img_path).convert('L')
            img = img.resize((105, 105))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
            
            support_images.append(img_tensor)
            support_labels.append(label_idx)
            support_display.append(img)
    
    if support_images:
        support_images = torch.cat(support_images, dim=0)
        support_labels = torch.tensor(support_labels)
        return support_images, support_labels, support_display, class_names
    else:
        return None, None, None, None

# ============================================
# PREDICT FUNCTION
# ============================================

def predict_image(model, uploaded_image, support_images, support_labels, class_names):
    """
    Predict which class the uploaded image belongs to
    """
    # Preprocess uploaded image
    img = uploaded_image.convert('L')
    img = img.resize((105, 105))
    img_array = np.array(img, dtype=np.float32) / 255.0
    query_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    
    # Run model
    with torch.no_grad():
        logits = model(support_images, support_labels, query_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        # Get all class probabilities
        all_probs = {class_names[i]: probs[0][i].item() for i in range(len(class_names))}
    
    return class_names[pred_class], confidence, all_probs

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📊 Results Summary")
st.sidebar.metric("1-Shot Accuracy", "89.68%", "±8.66%")
st.sidebar.metric("5-Shot Accuracy", "95.28%", "±5.12%")
st.sidebar.metric("Training Loss", "0.0894", "Excellent!")

st.sidebar.markdown("---")
st.sidebar.markdown("### How It Works")
st.sidebar.markdown("""
1. **Pre-trained on Greek** - Learned strokes, curves, patterns
2. **Meta-trained on Indus Valley** - Learned how to compare symbols  
3. **Recognizes new symbols** - From just 1-5 examples!
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
- **Model**: Prototypical Networks
- **Dataset**: MahadevanData (Indus Valley)
- **Pre-training**: Omniglot (Greek)
""")

# ============================================
# MAIN CONTENT
# ============================================

tab1, tab2, tab3 = st.tabs(["🔍 Test Your Own Image", "📊 Training Results", "ℹ️ About"])

# ============================================
# TAB 1: TEST YOUR OWN IMAGE
# ============================================

with tab1:
    st.header("Test with Your Own Indus Valley Symbol")
    
    # Load model
    model = load_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload an Indus Valley symbol image", type=['png', 'jpg', 'jpeg'])
        
        n_shot = st.selectbox("Number of Examples (Shot)", [1, 3, 5], index=2, 
                              help="More examples = better accuracy")
        
        # Refresh button
        if st.button("🔄 Refresh Support Examples", use_container_width=True):
            st.session_state.refresh_counter = st.session_state.get('refresh_counter', 0) + 1
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Sample Support Examples")
        st.markdown(f"These are {n_shot} real examples the model learns from:")
        
        # Load support examples (fresh each time due to refresh)
        support_images, support_labels, support_display, class_names = load_support_examples(n_shot)
        
        if support_display:
            # Display support images
            support_cols = st.columns(n_shot)
            for i, col in enumerate(support_cols):
                if i < len(support_display):
                    col.image(support_display[i], caption=f"Class: {class_names[i]}", use_container_width=True)
                else:
                    col.image("https://via.placeholder.com/80x80?text=Example", caption="Loading...")
        else:
            st.error("❌ No support examples found. Please check your dataset path.")
            st.info(f"Expected path: `{TEST_PATH}`")
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### Your Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, width=200)
            
            st.markdown("### Prediction Result")
            st.markdown("---")
            
            # Make prediction
            if support_images is not None and class_names:
                predicted_class, confidence, all_probs = predict_image(
                    model, image, support_images, support_labels, class_names
                )
                
                # Display confidence gauge
                if confidence >= 0.8:
                    st.success(f"### 🎯 Predicted Symbol: **{predicted_class}**")
                    st.metric("Confidence", f"{confidence*100:.1f}%", delta="High Confidence", delta_color="normal")
                elif confidence >= 0.5:
                    st.info(f"### 🎯 Predicted Symbol: **{predicted_class}**")
                    st.metric("Confidence", f"{confidence*100:.1f}%", delta="Medium Confidence", delta_color="off")
                else:
                    st.warning(f"### 🎯 Predicted Symbol: **{predicted_class}**")
                    st.metric("Confidence", f"{confidence*100:.1f}%", delta="Low Confidence", delta_color="inverse")
                
                st.markdown("---")
                
                # Show all class probabilities
                st.markdown("### All Class Probabilities")
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                
                for class_name, prob in sorted_probs:
                    st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")
                
                st.markdown("---")
                st.markdown("### Why This Prediction?")
                st.markdown(f"""
                - Pattern matched closest to prototype of class **{predicted_class}**
                - Learned features from Greek (curves, strokes, patterns)
                - Compared with {len(class_names)} support examples above
                - Model confidence: **{confidence*100:.1f}%**
                """)
                
                # Add interpretation
                if confidence > 0.7:
                    st.success("✅ **High confidence** - The model is very sure about this prediction!")
                elif confidence > 0.4:
                    st.info("📊 **Moderate confidence** - The model is reasonably confident but there are similar alternatives.")
                else:
                    st.warning("⚠️ **Low confidence** - The symbol might be unclear or similar to multiple classes.")
            else:
                st.error("❌ Could not load support examples. Please check your dataset.")
        else:
            st.info("👈 Upload an Indus Valley symbol image to see prediction")
            st.markdown("")
            st.markdown("**Where to get test images:**")
            