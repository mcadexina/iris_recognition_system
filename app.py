import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from io import BytesIO
import base64

# Import our modules
from utils.preprocessing import preprocess_image, extract_iris_region
from models.cnn_model import load_cnn_model, extract_features, predict_class
from models.gabor_model import extract_features as gabor_features
from models.wavelet_model import extract_features as wavelet_features

# Constants
MODEL_PATH = "iris_cnn_model.h5"
CLASS_MAPPING_FILE = "class_mapping.json"
UPLOAD_DIR = "uploads"
IMG_SIZE = (64, 64)

# Make sure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# App configuration and styling
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4da6ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #0066cc;
        margin-top: 1rem;
    }
    .card {
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-box {
        background-color: #e6f2ff;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        flex: 1;
        min-width: 150px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load class mapping
def load_class_mapping():
    try:
        with open(CLASS_MAPPING_FILE, 'r') as f:
            mapping = json.load(f)
        return mapping
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Class mapping file not found or invalid. Using generic class names.")
        # Return a generic mapping
        class_names = [f"Person{i+1}" for i in range(3)]
        mapping = {
            'class_to_idx': {name: i for i, name in enumerate(class_names)},
            'idx_to_class': {str(i): name for i, name in enumerate(class_names)}
        }
        return mapping

# Function to display an image with a downloadable link
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Process the uploaded image
def process_image(uploaded_file, use_preprocessing=True):
    try:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Check if the image is grayscale or RGB
        if image.mode == 'L':
            st.info("Detected grayscale image (mode 'L')")
            # Keep as grayscale, don't convert to RGB
            is_grayscale = True
        else:
            # For all other modes (RGB, RGBA, etc.), convert to RGB for consistent processing
            image = image.convert('RGB')
            is_grayscale = False
            st.info(f"Detected color image (converted from mode '{image.mode}' to 'RGB')")
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Add debug info about image shape
        st.info(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
        
        # Save the original image to the uploads folder
        timestamp = int(time.time())
        original_path = os.path.join(UPLOAD_DIR, f"original_{timestamp}.png")
        image.save(original_path)
        
        # Create a 2x2 grid of images showing processing steps
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
              # Extract iris region if requested
        if use_preprocessing:
            try:
                # Convert to grayscale for iris detection (iris detection works better on grayscale)
                if image_np.ndim == 3 and image_np.shape[2] == 3:
                    st.info("Converting to grayscale for iris detection")
                    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                else:
                    # Already grayscale
                    gray_image = image_np.copy()
                    
                # Extract iris region
                st.info("Attempting to extract iris region")
                iris_region = extract_iris_region(gray_image)
                
                if iris_region is not None:
                    st.success("Iris region successfully extracted")
                    axes[0, 1].imshow(iris_region, cmap='gray')
                    axes[0, 1].set_title("Extracted Iris Region")
                    
                    # Save the extracted iris
                    iris_path = os.path.join(UPLOAD_DIR, f"iris_{timestamp}.png")
                    cv2.imwrite(iris_path, iris_region)
                    
                    # Determine whether to keep color for preprocessing (if original was color)
                    keep_color = (image_np.ndim == 3 and image_np.shape[2] == 3)
                    
                    # Use the extracted iris for further processing - keep original color format if needed
                    st.info(f"Preprocessing iris region (keep_color={keep_color})")
                    processed_img = preprocess_image(iris_region, target_size=IMG_SIZE, enhance=True, keep_color=keep_color)
                else:
                    st.warning("Could not detect iris region. Using the whole image instead.")
                    # Process the original image with its original color format
                    keep_color = (image_np.ndim == 3 and image_np.shape[2] == 3)
                    st.info(f"Preprocessing whole image (keep_color={keep_color})")
                    processed_img = preprocess_image(image_np, target_size=IMG_SIZE, enhance=True, keep_color=keep_color)
                    
                    # Display grayscale version of the image for visualization
                    if image_np.ndim == 3 and image_np.shape[2] == 3:
                        axes[0, 1].imshow(gray_image, cmap='gray')
                    else:
                        axes[0, 1].imshow(image_np, cmap='gray')
                    axes[0, 1].set_title("Grayscale Image (No Iris Detected)")
            except Exception as e:
                st.warning(f"Error in iris extraction: {str(e)}. Using standard preprocessing.")
                # Log detailed error for debugging
                import traceback
                st.info(f"Error details: {traceback.format_exc()}")
                
                # Process the original image with its original color format
                keep_color = (image_np.ndim == 3 and image_np.shape[2] == 3)
                st.info(f"Preprocessing whole image (keep_color={keep_color})")
                processed_img = preprocess_image(image_np, target_size=IMG_SIZE, enhance=True, keep_color=keep_color)
                
                # Display grayscale version of the image for visualization
                if image_np.ndim == 3 and image_np.shape[2] == 3:
                    axes[0, 1].imshow(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY), cmap='gray')
                else:
                    axes[0, 1].imshow(image_np, cmap='gray')
                axes[0, 1].set_title("Grayscale Image")
        else:
            # Standard preprocessing without iris extraction - keep original color format
            keep_color = (image_np.ndim == 3 and image_np.shape[2] == 3)
            st.info(f"Standard preprocessing without iris extraction (keep_color={keep_color})")
            processed_img = preprocess_image(image_np, target_size=IMG_SIZE, enhance=True, keep_color=keep_color)
            
            # Display grayscale version of the image for visualization
            if image_np.ndim == 3 and image_np.shape[2] == 3:
                axes[0, 1].imshow(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY), cmap='gray')
            else:
                axes[0, 1].imshow(image_np, cmap='gray')
            axes[0, 1].set_title("Grayscale Image")
        
        # Show processed image - handle both grayscale and color images for display
        try:
            # Check if processed image is color (3D with 3 channels) or grayscale
            if processed_img.ndim == 3 and processed_img.shape[2] == 3:
                # Display as color image
                axes[1, 0].imshow(processed_img)
                axes[1, 0].set_title("Preprocessed Image (Color)")
            else:
                # Display as grayscale (with explicit cmap)
                if processed_img.ndim == 3:  # Has extra channel dimension
                    # If it's 3D but only has 1 channel, squeeze it
                    display_img = np.squeeze(processed_img)
                else:
                    display_img = processed_img
                    
                axes[1, 0].imshow(display_img, cmap='gray')
                axes[1, 0].set_title("Preprocessed Image (Grayscale)")
                
            axes[1, 0].axis('off')
        except Exception as display_error:
            st.warning(f"Error displaying processed image: {display_error}")
            # Fallback display method
            axes[1, 0].text(0.5, 0.5, "Error displaying image", 
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Empty space for feature visualization
        axes[1, 1].set_title("Feature Visualization")
        axes[1, 1].axis('off')
        
        # Save the figure for display
        plt.tight_layout()
        fig_path = os.path.join(UPLOAD_DIR, f"processing_{timestamp}.png")
        plt.savefig(fig_path)
        
        return image_np, processed_img, fig, fig_path, timestamp
        
    except Exception as e:
        # Global exception handling for the entire function
        st.error(f"Error processing image: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        
        # Create a default figure with error message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error processing image:\n{str(e)}", 
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        
        # Generate a timestamp for the error case
        timestamp = int(time.time())
        fig_path = os.path.join(UPLOAD_DIR, f"error_{timestamp}.png")
        plt.savefig(fig_path)
        
        # Return minimal values to avoid crashes downstream
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        return dummy_img, dummy_img, fig, fig_path, timestamp

# Visualize feature vectors (for CNN)
def visualize_features(features, fig_path):
    # Take first 100 features if more than 100
    features_to_show = features[:100]
    
    # Create a visualization of the feature vector
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(features_to_show)), features_to_show)
    plt.title("CNN Feature Vector Visualization")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    
    return fig_path

# Function for comparing features between images
def compare_features(features1, features2):
    # Calculate cosine similarity
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity

# Load model and make prediction
def predict_identity(img_array, model_choice="CNN"):
    try:
        if model_choice == "CNN":
            # Load CNN model
            st.info("Loading CNN model...")
            try:
                model = load_cnn_model(model_path=MODEL_PATH, debug=True)
                
                # Get the input shape information safely
                try:
                    input_shape = None
                    
                    # Try multiple approaches to determine input shape
                    if hasattr(model, 'input_shape') and model.input_shape is not None:
                        input_shape = model.input_shape
                        st.info(f"Model expects input shape from model.input_shape: {input_shape}")
                    # Check first layer's _batch_input_shape
                    elif hasattr(model.layers[0], '_batch_input_shape') and model.layers[0]._batch_input_shape is not None:
                        input_shape = model.layers[0]._batch_input_shape
                        st.info(f"Model expects input shape from first layer _batch_input_shape: {input_shape}")
                    # Try to infer from model.inputs
                    elif hasattr(model, 'inputs') and model.inputs:
                        input_shape = model.inputs[0].shape
                        st.info(f"Model expects input shape from model.inputs: {input_shape}")
                    # Default fallback
                    else:
                        input_shape = (None, 64, 64, 1)
                        st.warning(f"Could not determine model input shape, using default: {input_shape}")
                        
                    # Create properly preprocessed image based on model's input shape
                    if input_shape and len(input_shape) > 1:
                        expected_height = input_shape[1] if input_shape[1] is not None else 64
                        expected_width = input_shape[2] if input_shape[2] is not None else 64
                        st.info(f"Preprocessing image to size: {expected_height}x{expected_width}")
                        
                        # Update the IMG_SIZE to match the model's expected input
                        global IMG_SIZE
                        IMG_SIZE = (expected_height, expected_width)
                except Exception as shape_error:
                    st.warning(f"Error determining input shape: {str(shape_error)}")
                    st.info("Using default image size: (64, 64)")
            except Exception as e:
                st.error(f"Error loading CNN model: {str(e)}")
                st.warning("Will attempt to create a new model...")
                try:
                    from models.cnn_model import create_cnn_model
                    model = create_cnn_model(input_shape=(64, 64, 1), num_classes=3)
                    st.warning("Created new model. Note: This model is not trained and will not give accurate predictions.")
                except Exception as create_error:
                    st.error(f"Failed to create new model: {str(create_error)}")
                    return None, None
            
            # Extract features and predict using the adapted image size
            st.info("Extracting features...")
            features = extract_features(img_array, model)
            
            # Make prediction
            st.info("Making prediction...")
            prediction = predict_class(img_array, model)
            
            # Get class mapping
            mapping = load_class_mapping()
            idx_to_class = mapping['idx_to_class']
              # Get prediction probability - use the CNN's preprocess_image function for proper tensor formatting
            from models.cnn_model import preprocess_image as cnn_preprocess
            
            # Get model's expected input shape and channels
            expected_channels = 1  # Default to grayscale
            
            # Try to determine if the model expects color (3 channels) or grayscale (1 channel)
            try:
                if model.input_shape and len(model.input_shape) > 3:
                    expected_channels = model.input_shape[3]
                    st.info(f"Model expects {expected_channels} channel(s)")
            except Exception as e:
                st.warning(f"Could not determine expected channels, defaulting to grayscale: {str(e)}")
            
            # Process the image specifically for model prediction
            try:
                # First, ensure we have the right type of image (color or grayscale)
                if expected_channels == 3 and (len(img_array.shape) == 2 or img_array.shape[2] == 1):
                    st.info("Converting grayscale to RGB for model prediction")
                    if len(img_array.shape) == 2:
                        img_array_for_prediction = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    else:
                        img_array_for_prediction = cv2.cvtColor(img_array[:,:,0], cv2.COLOR_GRAY2RGB)
                elif expected_channels == 1 and len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    st.info("Converting RGB to grayscale for model prediction")
                    img_array_for_prediction = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    img_array_for_prediction = img_array.copy()
                
                # Use the CNN's preprocessing function which returns a properly formatted tensor
                processed_image = cnn_preprocess(img_array_for_prediction, target_size=IMG_SIZE)
                
                # Debug info about the processed image
                st.info(f"Processed image shape before prediction: {processed_image.shape}")
                
                # Make prediction with the properly formatted tensor
                probabilities = model.predict(processed_image, verbose=0)[0]
            except Exception as predict_error:
                st.error(f"Error during prediction: {str(predict_error)}")
                import traceback
                st.error(f"Prediction error details: {traceback.format_exc()}")
                # Return dummy probabilities for graceful error handling
                probabilities = np.zeros(3)
                probabilities[0] = 1.0  # Assign 100% to first class as a fallback
            
            # Ensure prediction is valid
            if prediction >= len(probabilities):
                prediction = 0  # Default to first class if prediction is invalid
                
            confidence = float(probabilities[prediction]) * 100
            predicted_class = idx_to_class.get(str(prediction), f"Person{prediction+1}")
            
            return {
                "features": features,                "prediction": prediction,
                "class_name": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities.tolist()
            }
            
        elif model_choice == "Gabor":
            st.info("Extracting Gabor filter features...")
            try:
                features = gabor_features(img_array)
                st.success("Gabor feature extraction complete")
                return {"features": features, "prediction": None, "class_name": None, "confidence": None, "probabilities": []}
            except Exception as e:
                st.error(f"Error in Gabor feature extraction: {str(e)}")
                return {"features": np.zeros(100), "prediction": None, "class_name": "Error", "confidence": None, "probabilities": []}
            
        elif model_choice == "Wavelet":
            st.info("Extracting Wavelet transform features...")
            try:
                features = wavelet_features(img_array)
                st.success("Wavelet feature extraction complete")
                return {"features": features, "prediction": None, "class_name": None, "confidence": None, "probabilities": []}
            except Exception as e:
                st.error(f"Error in Wavelet feature extraction: {str(e)}")
                return {"features": np.zeros(200), "prediction": None, "class_name": "Error", "confidence": None, "probabilities": []}
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        # Return minimal result with empty features to avoid crashes
        return {"features": np.zeros(128), "prediction": None, "class_name": "Error", "confidence": 0, "probabilities": []}

# Sidebar navigation
def sidebar_menu():
    st.sidebar.image("https://img.freepik.com/premium-vector/drawing-blue-eye-with-blue-eye-it_1187092-26294.jpg", width=250)
    st.sidebar.title("Iris Recognition System")
    
    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "Recognition", "About"]
    )
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("System Status")
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        st.sidebar.success("✅ CNN Model Loaded")
        
        # Add option to rebuild model if there are compatibility issues
        if st.sidebar.button("Rebuild Model"):
            try:
                st.sidebar.info("Creating new model...")
                from models.cnn_model import create_cnn_model
                model = create_cnn_model(input_shape=(64, 64, 1), num_classes=3)
                model.save(MODEL_PATH)
                st.sidebar.success("✅ Model rebuilt successfully!")
            except Exception as e:
                st.sidebar.error(f"Error rebuilding model: {str(e)}")
    else:
        st.sidebar.error("❌ CNN Model Not Found")
        
        # Add option to create the model
        if st.sidebar.button("Create New Model"):
            try:
                st.sidebar.info("Creating new model...")
                from models.cnn_model import create_cnn_model
                model = create_cnn_model(input_shape=(64, 64, 1), num_classes=3)
                model.save(MODEL_PATH)
                st.sidebar.success("✅ Model created successfully!")
            except Exception as e:
                st.sidebar.error(f"Error creating model: {str(e)}")
    
    # Check if class mapping exists
    if os.path.exists(CLASS_MAPPING_FILE):
        st.sidebar.success("✅ Class Mapping Loaded")
    else:
        st.sidebar.warning("⚠️ Using Default Class Mapping")
        
        # Add option to create a default class mapping
        if st.sidebar.button("Create Default Class Mapping"):
            try:
                class_names = [f"Person{i+1}" for i in range(3)]
                mapping = {
                    'class_to_idx': {name: i for i, name in enumerate(class_names)},
                    'idx_to_class': {str(i): name for i, name in enumerate(class_names)}
                }
                with open(CLASS_MAPPING_FILE, 'w') as f:
                    json.dump(mapping, f)
                st.sidebar.success("✅ Default class mapping created!")
            except Exception as e:
                st.sidebar.error(f"Error creating class mapping: {str(e)}")
      # Add an advanced debug option
    with st.sidebar.expander("Debug Options"):
        if st.button("Show Model Info"):
            try:
                from tensorflow.keras.models import load_model
                model = load_model(MODEL_PATH)
                
                # Show basic model info
                st.write("### Basic Model Information")
                st.write(f"Model input shape: {model.input_shape}")
                st.write(f"First layer input shape: {model.layers[0].input_shape}")
                st.write(f"Last layer output shape: {model.layers[-1].output_shape}")
                
                # Determine if model expects color or grayscale
                expected_channels = None
                if model.input_shape and len(model.input_shape) > 3:
                    expected_channels = model.input_shape[3]
                    
                if expected_channels == 1:
                    st.info("This model expects GRAYSCALE images (1 channel)")
                    st.markdown("""
                    Color images will be automatically converted to grayscale.
                    """)
                elif expected_channels == 3:
                    st.info("This model expects COLOR images (3 channels/RGB)")
                    st.markdown("""
                    Grayscale images will be automatically converted to RGB.
                    """)
                else:
                    st.warning(f"Unexpected channel configuration: {expected_channels}")
                
                # Add a button to test image preprocessing
                if st.button("Test Image Preprocessing"):
                    # Create a sample grayscale and color image
                    gray_test = np.zeros((64, 64), dtype=np.uint8)
                    color_test = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    # Import the preprocessing function from CNN model
                    from models.cnn_model import preprocess_image as cnn_preprocess
                    
                    # Test preprocessing both images
                    gray_processed = cnn_preprocess(gray_test)
                    color_processed = cnn_preprocess(color_test)
                    
                    # Show results
                    st.write(f"Grayscale image processed shape: {gray_processed.shape}")
                    st.write(f"Color image processed shape: {color_processed.shape}")
                    
                    # Test if model can accept both
                    try:
                        gray_pred = model.predict(gray_processed, verbose=0)
                        st.success("Model successfully accepted grayscale image")
                    except Exception as e:
                        st.error(f"Error with grayscale image: {e}")
                        
                    try:
                        color_pred = model.predict(color_processed, verbose=0)
                        st.success("Model successfully accepted color image")
                    except Exception as e:
                        st.error(f"Error with color image: {e}")
                
                # Layer summary
                st.write("### Layer Details")
                layers_info = []
                for i, layer in enumerate(model.layers):
                    if hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape'):
                        layers_info.append({
                            "Layer": i,
                            "Name": layer.name,
                            "Type": layer.__class__.__name__,
                            "Input Shape": str(layer.input_shape),
                            "Output Shape": str(layer.output_shape)
                        })
                
                st.table(layers_info)
            except Exception as e:
                st.error(f"Error loading model info: {str(e)}")
    
    return menu

# Home page
def show_home():
    st.markdown('<h1 class="main-header">Iris Recognition System</h1>', unsafe_allow_html=True)
    
    # About the system
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Welcome</h2>', unsafe_allow_html=True)
    st.markdown("""
    This system uses advanced deep learning techniques to recognize individuals 
    based on their iris patterns. The iris is one of the most unique biometric 
    identifiers with over 200 distinct features.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3>Key Features</h3>', unsafe_allow_html=True)
        st.markdown("""
        - 👁 Iris region extraction
        - 🧠 Deep CNN feature extraction
        - 🔍 Advanced image preprocessing
        - 📊 Multiple feature extraction methods
        - 🚀 Fast and accurate identification
        """)
    
    with col2:
        st.markdown('<h3>How to Use</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. Navigate to the Recognition page
        2. Upload an iris image
        3. Choose a feature extraction method
        4. View the prediction results
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Demo image
    st.image("https://www.researchgate.net/publication/328189177/figure/fig3/AS:678307209416707@1538746510484/Iris-Recognition-System.png")
    
    # Get started button
    if st.button("Get Started with Recognition", key="home_button"):
        st.session_state.menu = "Recognition"
        st.rerun()

# Recognition page
def show_recognition():
    st.markdown('<h1 class="main-header">Iris Recognition</h1>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Upload Iris Image</h2>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        model_choice = st.selectbox(
            "Choose a model:",
            ["CNN", "Gabor", "Wavelet"],
            help="CNN is recommended for best accuracy"
        )
    
    with col2:
        use_preprocessing = st.checkbox("Extract Iris Region", value=True,
                                      help="Automatically detect and extract the iris region")
    
    with col3:
        show_features = st.checkbox("Visualize Features", value=True,
                                  help="Show feature vector visualization")
    
    uploaded_file = st.file_uploader("Choose an iris image...", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process the image when uploaded
    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            # Process the image
            image_np, processed_img, fig, fig_path, timestamp = process_image(
                uploaded_file, use_preprocessing)
              # Show the processing steps
            st.image(fig_path, caption="Image Processing Steps", use_container_width=True)
            
            # Feature extraction and prediction
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="subheader">Recognition Results</h2>', unsafe_allow_html=True)
            
            with st.spinner(f"Extracting features using {model_choice} model..."):
                result = predict_identity(image_np, model_choice)
                
                if result and "features" in result:
                    # Show feature extraction success
                    st.markdown(f'<div class="success-box">Features successfully extracted using {model_choice} model</div>', unsafe_allow_html=True)
                      # Display feature visualization if requested
                    if show_features and result["features"] is not None:
                        try:
                            feature_viz_path = os.path.join(UPLOAD_DIR, f"features_{timestamp}.png")
                            visualize_features(result["features"], feature_viz_path)
                            st.image(feature_viz_path, caption="Feature Visualization", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not visualize features: {str(e)}")
                    
                    # Show prediction results for CNN
                    if model_choice == "CNN" and result["class_name"]:
                        st.markdown('<h3>Prediction Results</h3>', unsafe_allow_html=True)
                          # Create metrics display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Identity", result["class_name"])
                            st.metric("Confidence", f"{result['confidence']:.2f}%")
                        
                        with col2:
                            try:
                                # Display probabilities as a bar chart
                                mapping = load_class_mapping()
                                idx_to_class = mapping['idx_to_class']
                                
                                # Make sure probabilities exist
                                if result["probabilities"] and len(result["probabilities"]) > 0:
                                    class_names = [idx_to_class.get(str(i), f"Person{i+1}") 
                                                for i in range(len(result["probabilities"]))]
                                    
                                    # Create probability dataframe
                                    prob_df = pd.DataFrame({
                                        'Class': class_names,
                                        'Probability': result["probabilities"]
                                    })
                                    
                                    # Sort by probability
                                    prob_df = prob_df.sort_values('Probability', ascending=False)
                                    
                                    # Show top 5 probabilities (or fewer if there are less than 5)
                                    num_to_show = min(5, len(prob_df))
                                    if num_to_show > 0:
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        sns.barplot(x='Probability', y='Class', data=prob_df.head(num_to_show), ax=ax)
                                        ax.set_title('Top Class Probabilities')
                                        ax.set_xlabel('Probability')
                                        ax.set_ylabel('Class')
                                        st.pyplot(fig)
                                    else:
                                        st.warning("No probability data available to display.")
                                else:
                                    st.warning("No probability data available to display.")
                            except Exception as e:
                                st.warning(f"Could not display probability chart: {str(e)}")
                    else:
                        # For non-CNN models, just show feature info
                        st.info(f"Feature vector length: {len(result['features'])}")
                        st.write("Feature sample (first 10 values):", result["features"][:10])
                        
                        st.warning("Note: Gabor and Wavelet methods extract features but don't directly predict identity. "
                                  "Use the CNN model for full identification.")
                
            st.markdown('</div>', unsafe_allow_html=True)

# About page
def show_about():
    st.markdown('<h1 class="main-header">About the Iris Recognition System</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## How the System Works

    This iris recognition system uses a combination of image processing techniques and 
    deep learning models to identify individuals based on their unique iris patterns.
    
    ### Processing Pipeline:

    1. **Image Input**: System accepts an iris image upload
    2. **Preprocessing**: 
        - Grayscale conversion
        - Iris region extraction using Hough Circle Transform
        - Image enhancement (CLAHE, denoising)
        - Resizing to standardized dimensions
    3. **Feature Extraction**: Multiple methods available
        - CNN (Convolutional Neural Network): Deep learning approach
        - Gabor Filter: Texture-based feature extraction
        - Wavelet Transform: Frequency domain analysis
    4. **Classification**: For CNN model, predicts the identity from trained classes
    
    ### Technical Implementation:

    - The CNN model has been trained on a dataset of iris images
    - The model architecture includes multiple convolutional layers with batch normalization
    - Key evaluation metrics include accuracy, precision, and recall
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## References
    
    1. Daugman, J. (2009). How iris recognition works. *The essential guide to image processing*
    2. Bowyer, K. W., Hollingsworth, K., & Flynn, P. J. (2008). Image understanding for iris biometrics: A survey. *Computer vision and image understanding*
    3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="footer">Iris Recognition System v1.0 | © 2025</p>', unsafe_allow_html=True)

# Main app function
def main():
    apply_custom_css()
    
    # Initialize session state for navigation
    if "menu" not in st.session_state:
        st.session_state.menu = "Home"
    
    # Get menu selection from sidebar
    menu = sidebar_menu()
    
    # Update session state if menu changed
    if menu != st.session_state.menu:
        st.session_state.menu = menu
    
    # Show the appropriate page based on navigation
    if st.session_state.menu == "Home":
        show_home()
    elif st.session_state.menu == "Recognition":
        show_recognition()
    elif st.session_state.menu == "About":
        show_about()

# Run the app
if __name__ == "__main__":
    main()
