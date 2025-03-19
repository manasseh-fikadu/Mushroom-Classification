import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile
import numpy as np
import time
import os

# For Lightning models
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score

# Ensure truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set page configuration
st.set_page_config(
    page_title="Mushroom Classification App",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the EfficientNet model class
class MushroomClassifierEfficientNet(pl.LightningModule):
    def __init__(self, num_classes, efficientnet_version='b4', dropout_rate=0.3):
        super().__init__()
        
        # Load the selected EfficientNet model
        if efficientnet_version == 'b0':
            self.backbone = models.efficientnet_b0(weights=None)
            feature_dim = 1280
        elif efficientnet_version == 'b1':
            self.backbone = models.efficientnet_b1(weights=None)
            feature_dim = 1280
        elif efficientnet_version == 'b2':
            self.backbone = models.efficientnet_b2(weights=None)
            feature_dim = 1408
        elif efficientnet_version == 'b3':
            self.backbone = models.efficientnet_b3(weights=None)
            feature_dim = 1536
        elif efficientnet_version == 'b4':
            self.backbone = models.efficientnet_b4(weights=None)
            feature_dim = 1792
        else:
            raise ValueError(f"Unsupported EfficientNet version: {efficientnet_version}")
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add a spatial attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Add a new classifier with dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Save feature dimension for forward pass
        self.feature_dim = feature_dim
        
        # Add these attributes to match the saved model
        self.train_accuracy = None
        self.val_accuracy = None
        self.val_precision = None
        self.val_recall = None
        self.val_f1 = None
        self.test_accuracy = None
        self.test_precision = None
        self.test_recall = None
        self.test_f1 = None
        self.class_weights = None
        self.outputs = []
        self.train_losses = []
        self.val_losses = []
        self._epoch_counter = 0
        self.unfreeze_strategy = 'progressive'
        self.efficientnet_version = efficientnet_version
    
    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone.features(x)
        
        # Apply attention mechanism
        attention_map = self.attention(features)
        attended_features = features * attention_map
        
        # Pass through the classifier
        return self.classifier(attended_features)

# Define class names
class_names = [
    "Agaricus", "Amanita", "Boletus", "Cortinarius", 
    "Entoloma", "Hygrocybe", "Lactarius", "Russula", "Suillus"
]

# Define the normalization values
mean = [0.5, 0.5, 0.5]  # Using standard normalization as fallback
std = [0.25, 0.25, 0.25]  # Using standard normalization as fallback

# Define the image transformation pipeline
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# Function to load model
def load_model(model_path):
    num_classes = len(class_names)
    model = MushroomClassifierEfficientNet(num_classes, efficientnet_version='b4')
    
    # Load the model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Handle different types of saved models
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # This is a Lightning checkpoint
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if it exists in the state dict keys
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        # This is a direct state dict
        # Try to load with strict=False to ignore missing keys
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model

# Function to get class information
def get_class_info(class_name):
    class_info = {
        "Agaricus": {
            "description": "This genus includes the common mushroom, which is widely cultivated and consumed around the world. It is also found in the wild and can be identified by its white cap and pink gills.",
            "edibility": "Some species are edible, while others are poisonous.",
            "color": "#8BC34A"  # Light green
        },
        "Amanita": {
            "description": "This genus includes some of the most toxic mushrooms known to man, such as the death cap and destroying angel. They are characterized by their distinctive cup-shaped volva at the base of the stem and their white spores.",
            "edibility": "Some species are edible, but many are HIGHLY TOXIC and can be DEADLY. Extreme caution is required.",
            "color": "#F44336"  # Red
        },
        "Boletus": {
            "description": "This genus includes several edible species, such as the cep or porcini mushroom, which is highly prized for its rich flavor and meaty texture. Boletus mushrooms have a distinctive cap that is often brown or reddish-brown in color and a porous underside instead of gills.",
            "edibility": "Many species are edible and prized for cooking.",
            "color": "#795548"  # Brown
        },
        "Cortinarius": {
            "description": "This genus includes many species that are difficult to identify and some that are poisonous. They are characterized by their rusty brown spores and the remnants of a veil that covers the stem when young.",
            "edibility": "Some species are edible, but many are not recommended for consumption due to toxicity risks.",
            "color": "#FF9800"  # Orange
        },
        "Entoloma": {
            "description": "This genus includes many species that are difficult to identify and some that are poisonous. They are characterized by their pink or lilac spores and their gills that are attached to the stem instead of running down it.",
            "edibility": "Some species are edible, but many are not recommended for consumption.",
            "color": "#9C27B0"  # Purple
        },
        "Hygrocybe": {
            "description": "This genus includes many species that are brightly colored and often found in grassy areas or mossy woods. They are characterized by their waxy caps and their gills that are usually brightly colored as well.",
            "edibility": "Some species are edible, but many have not been tested for toxicity.",
            "color": "#FFEB3B"  # Yellow
        },
        "Lactarius": {
            "description": "This genus includes many species that exude a milky substance when cut or broken. They are characterized by their gills that often have a decurrent attachment to the stem and their white spores.",
            "edibility": "Some species are edible, but others can cause gastrointestinal distress.",
            "color": "#FF5722"  # Deep orange
        },
        "Russula": {
            "description": "This genus includes many species that have brightly colored caps and white spores. They are characterized by their brittle flesh and their gills that do not run down the stem.",
            "edibility": "Some species are edible, but others can cause gastrointestinal distress.",
            "color": "#E91E63"  # Pink
        },
        "Suillus": {
            "description": "This genus includes several edible species, such as the slippery jack mushroom, which is often found in coniferous forests. Suillus mushrooms have a distinctive slimy cap and pores instead of gills.",
            "edibility": "Generally edible, though some people may have adverse reactions.",
            "color": "#FFC107"  # Amber
        }
    }
    
    return class_info.get(class_name, {"description": "No information available.", "edibility": "Unknown", "color": "#9E9E9E"})

# Function to make predictions
def predict(model, image):
    # Apply transformations
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Get model predictions
        start_time = time.time()
        outputs = model(image_tensor)
        inference_time = time.time() - start_time
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top predictions
        topk_prob, topk_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(len(topk_indices)):
            idx = topk_indices[i].item()
            prob = topk_prob[i].item() * 100
            predictions.append((class_names[idx], prob))
    
    return predictions, inference_time

# Main app
def main():
    st.title("🍄 Mushroom Classification App")
    
    # Add sidebar for model information
    st.sidebar.title("Model Information")
    
    # Load the model
    model_path = "efficientnet_model.pth"
    model = load_model(model_path)
    st.sidebar.info("EfficientNet-B4 model loaded successfully!")
    
    # Add information about the model
    with st.sidebar.expander("About the Model"):
        st.write("""
        - **EfficientNet-B4**: A model that balances network depth, width, and resolution for better efficiency and accuracy.
        """)
    
    # Add information about the mushroom classes
    with st.sidebar.expander("About Mushroom Classes"):
        st.write("""
        This app can classify 9 different genera of mushrooms commonly found in Northern Europe:
        - Agaricus
        - Amanita
        - Boletus
        - Cortinarius
        - Entoloma
        - Hygrocybe
        - Lactarius
        - Russula
        - Suillus
        
        **IMPORTANT**: This app is for educational purposes only. Never consume wild mushrooms based solely on an app's identification.
        """)
    
    # Create two columns for the main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose a mushroom image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("")
            
            # Add a button to trigger prediction
            if st.button("Classify Mushroom"):
                with st.spinner("Classifying..."):
                    # Make prediction
                    predictions, inference_time = predict(model, image)
                    
                    # Display results in the second column
                    with col2:
                        st.header("Classification Results")
                        st.write(f"Inference time: {inference_time:.4f} seconds")
                        
                        # Display top prediction with a large colored box
                        top_class, top_prob = predictions[0]
                        class_info = get_class_info(top_class)
                        
                        st.markdown(f"""
                        <div style="background-color:{class_info['color']}; padding:20px; border-radius:10px; margin-bottom:20px;">
                            <h2 style="color:white; margin:0;">{top_class}</h2>
                            <h3 style="color:white; margin:0; opacity:0.9;">{top_prob:.1f}% confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display class information
                        st.subheader("About this mushroom genus:")
                        st.write(class_info["description"])
                        
                        # Display edibility with appropriate styling
                        st.subheader("Edibility:")
                        if "TOXIC" in class_info["edibility"] or "DEADLY" in class_info["edibility"]:
                            st.error(class_info["edibility"])
                        elif "edible" in class_info["edibility"].lower():
                            st.success(class_info["edibility"])
                        else:
                            st.warning(class_info["edibility"])
                        
                        # Display warning
                        st.warning("⚠️ IMPORTANT: Never consume wild mushrooms based solely on an app's identification. Always consult with a professional mycologist.")
                        
                        # Display other predictions
                        st.subheader("Other possibilities:")
                        for class_name, prob in predictions[1:]:
                            st.write(f"- {class_name}: {prob:.1f}%")
        else:
            # Display placeholder when no image is uploaded
            with col2:
                st.header("Classification Results")
                st.info("Upload an image to see classification results.")

if __name__ == "__main__":
    main()
