import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Import our custom model
from models.cnn_model import create_cnn_model

# CONFIG
DATA_DIR = "dataset"  # Your dataset structure
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = "iris_cnn_model.h5"
CHECKPOINT_PATH = "checkpoints/model-{epoch:02d}-{val_accuracy:.4f}.h5"
LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CLASS_MAPPING_FILE = "class_mapping.json"

# Step 1: Load and preprocess data
def load_data(data_dir, img_size=(64, 64)):
    """Load and preprocess images from the dataset directory"""
    X, y, class_names = [], [], []
    
    # Get only valid class directories (person folders)
    valid_dirs = [d for d in sorted(os.listdir(data_dir)) 
                if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    # Skip any non-person directories
    class_names = [d for d in valid_dirs if d.startswith('person')]
    
    if not class_names:
        raise ValueError(f"No valid person directories found in {data_dir}")
        
    # Create mapping from class names to indices
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Load images for each class
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        print(f"Processing class {cls}...")
        
        # Get all image files
        img_files = [f for f in os.listdir(cls_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not img_files:
            print(f"Warning: No images found in {cls_path}")
            continue
            
        # Process each image
        for img_name in tqdm(img_files, desc=f"Loading {cls}"):
            img_path = os.path.join(cls_path, img_name)
            try:
                # Open and preprocess the image
                img = Image.open(img_path).convert("L")  # Grayscale
                img = img.resize(img_size)
                img = np.array(img) / 255.0
                X.append(img[..., np.newaxis])  # Add channel dimension
                y.append(class_to_idx[cls])
            except Exception as e:
                print(f"Skipped {img_path}: {e}")

    if not X:
        raise ValueError("No valid images were loaded")
    
    # Save class mapping
    with open(CLASS_MAPPING_FILE, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(i): cls for cls, i in class_to_idx.items()}
        }, f)
        
    return np.array(X), np.array(y), class_names

# Data augmentation for training
def create_data_generator():
    """Create data augmentation generator for training data"""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,  # Typically False for iris images
        fill_mode='nearest'
    )

# Visualization functions
def plot_training_history(history, save_path="training_history.png"):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")

def create_callbacks():
    """Create callbacks for model training"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # TensorBoard for visualization
    tensorboard = TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1
    )
    
    return [checkpoint, early_stopping, reduce_lr, tensorboard]

# Step 3: Train and save model
def main():
    print("[INFO] Loading dataset...")
    try:
        X, y, class_names = load_data(DATA_DIR, IMG_SIZE)
        print(f"[INFO] Loaded {len(X)} images across {len(class_names)} classes.")
        print(f"[INFO] Image shape: {X[0].shape}")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"[INFO] Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Create data generator for training data
        datagen = create_data_generator()
        datagen.fit(X_train)
        
        # Create model with correct input shape and number of classes
        print("[INFO] Building model...")
        model = create_cnn_model(input_shape=X[0].shape, num_classes=len(class_names))
        model.summary()
        
        # Create callbacks
        callbacks = create_callbacks()
        
        # Train the model with data augmentation
        print("[INFO] Training with data augmentation...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        print("[INFO] Evaluating model...")
        scores = model.evaluate(X_val, y_val, verbose=1)
        print(f"[INFO] Validation Loss: {scores[0]:.4f}")
        print(f"[INFO] Validation Accuracy: {scores[1]:.4f}")
        
        # Save the final model
        print("[INFO] Saving final model...")
        model.save(MODEL_PATH)
        print(f"[DONE] Model saved to {MODEL_PATH}")
        
        # Plot and save training history
        plot_training_history(history)
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    main()
