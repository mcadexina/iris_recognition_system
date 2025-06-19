import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import json
import seaborn as sns
from PIL import Image, ImageDraw
import random
from tqdm import tqdm

from utils.preprocessing import preprocess_image
from models.cnn_model import extract_features, predict_class

# CONFIG
MODEL_PATH = "iris_cnn_model.h5"
TEST_DIR = "dataset"  # Can be the same as training or a separate test set
CLASS_MAPPING_FILE = "class_mapping.json"
IMG_SIZE = (64, 64)
SAMPLE_VISUALIZATIONS = 10  # Number of sample predictions to visualize

def create_synthetic_iris(save_path, img_size=128, num_images=5):
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(num_images):
        # Create blank grayscale image
        img = Image.new('L', (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)
        
        # Draw concentric circles to mimic iris texture
        center = (img_size // 2, img_size // 2)
        max_radius = img_size // 2 - 5
        
        for r in range(max_radius, 10, -5):
            intensity = random.randint(50, 200)
            bbox = [center[0]-r, center[1]-r, center[0]+r, center[1]+r]
            draw.ellipse(bbox, outline=intensity)
            
            # Add some random lines/rays
            for _ in range(5):
                angle = random.uniform(0, 2*np.pi)
                length = random.randint(r//2, r)
                x_end = center[0] + int(length * np.cos(angle))
                y_end = center[1] + int(length * np.sin(angle))
                draw.line([center, (x_end, y_end)], fill=intensity, width=1)
        
        # Add noise
        noise = np.random.randint(0, 30, (img_size, img_size)).astype('uint8')
        img_arr = np.array(img)
        img_arr = np.clip(img_arr + noise, 0, 255).astype('uint8')
        img = Image.fromarray(img_arr)
        
        img.save(os.path.join(save_path, f'img_{i+1}.png'))

def generate_dataset(base_dir='synthetic_iris_dataset', persons=3, images_per_person=10):
    os.makedirs(base_dir, exist_ok=True)
    for p in range(1, persons+1):
        person_folder = os.path.join(base_dir, f'person{p}')
        create_synthetic_iris(person_folder, num_images=images_per_person)
    print(f"Synthetic iris dataset generated at '{base_dir}' with {persons} persons.")

# Load class mapping
def load_class_mapping():
    with open(CLASS_MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    return mapping

# Load test data
def load_test_data(test_dir, img_size=(64, 64)):
    X_test, y_test, filenames = [], [], []
    
    # Load mapping
    try:
        with open(CLASS_MAPPING_FILE, 'r') as f:
            mapping = json.load(f)
            class_to_idx = mapping['class_to_idx']
    except FileNotFoundError:
        print(f"Warning: Class mapping file {CLASS_MAPPING_FILE} not found.")
        # Create a simple mapping based on directory names
        valid_dirs = [d for d in sorted(os.listdir(test_dir)) 
                    if os.path.isdir(os.path.join(test_dir, d)) and d.startswith('person')]
        class_to_idx = {cls: i for i, cls in enumerate(valid_dirs)}
    
    # Get valid class directories (person folders)
    valid_dirs = [d for d in sorted(os.listdir(test_dir)) 
                if os.path.isdir(os.path.join(test_dir, d)) and d.startswith('person')]
                
    for cls in valid_dirs:
        cls_path = os.path.join(test_dir, cls)
        cls_idx = class_to_idx.get(cls)
        
        if cls_idx is None:
            print(f"Warning: Class {cls} not found in training set, skipping")
            continue
            
        # Get all image files
        img_files = [f for f in os.listdir(cls_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Process each image
        for img_name in img_files:
            img_path = os.path.join(cls_path, img_name)
            try:
                # Open and preprocess image
                img = Image.open(img_path).convert("L")  # Grayscale
                img = img.resize(img_size)
                img = np.array(img) / 255.0
                X_test.append(img[..., np.newaxis])  # Add channel dimension
                y_test.append(cls_idx)
                filenames.append(img_path)
            except Exception as e:
                print(f"Skipped {img_path}: {e}")
                
    return np.array(X_test), np.array(y_test), filenames

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png")

# Visualize some predictions
def visualize_predictions(X_test, y_test, y_pred, filenames, mapping, num_samples=10):
    idx_to_class = mapping['idx_to_class']
    
    # Select random samples
    indices = np.random.choice(range(len(X_test)), min(num_samples, len(X_test)), replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        img = X_test[idx].squeeze()  # Remove channel dimension
        true_class = idx_to_class[str(y_test[idx])]
        pred_class = idx_to_class[str(y_pred[idx])]
        
        # Display image
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_class}\nPred: {pred_class}")
        axes[i].axis('off')
        
        # Highlight correct/incorrect predictions
        if y_test[idx] == y_pred[idx]:
            for spine in axes[i].spines.values():
                spine.set_color('green')
                spine.set_linewidth(2)
        else:
            for spine in axes[i].spines.values():
                spine.set_color('red')
                spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.close()
    print("Sample predictions saved to prediction_samples.png")

def evaluate_model():
    try:
        print("[INFO] Loading model...")
        model = load_model(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
        
        print("[INFO] Loading class mapping...")
        try:
            mapping = load_class_mapping()
            idx_to_class = mapping['idx_to_class']
            class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Couldn't load class mapping: {e}")
            # Continue with generic class names
            class_names = [f"Person{i+1}" for i in range(3)]  # Default to 3 classes
            mapping = {
                'class_to_idx': {name: i for i, name in enumerate(class_names)},
                'idx_to_class': {str(i): name for i, name in enumerate(class_names)}
            }
        
        print("[INFO] Loading test data...")
        X_test, y_test, filenames = load_test_data(TEST_DIR, IMG_SIZE)
        print(f"[INFO] Loaded {len(X_test)} test images")
        
        # Evaluate model
        print("[INFO] Evaluating model...")
        scores = model.evaluate(X_test, y_test, verbose=1)
        print(f"[INFO] Test Loss: {scores[0]:.4f}")
        print(f"[INFO] Test Accuracy: {scores[1]:.4f}")
        
        # Get predictions
        print("[INFO] Generating predictions...")
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate classification report
        print("\n[INFO] Classification Report:")
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
        
        # Save report to file
        with open('classification_report.txt', 'w') as f:
            f.write(report)
            
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, class_names)
        
        # Visualize some predictions
        visualize_predictions(X_test, y_test, y_pred, filenames, mapping, SAMPLE_VISUALIZATIONS)
        
        print("[DONE] Model evaluation complete")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == '__main__':
    choice = input("Choose operation: \n1. Generate Synthetic Dataset\n2. Evaluate Model\nChoice (1/2): ")
    
    if choice == '1':
        persons = int(input("Number of persons to generate (default 3): ") or 3)
        images = int(input("Images per person (default 10): ") or 10)
        generate_dataset(persons=persons, images_per_person=images)
    elif choice == '2':
        evaluate_model()
    else:
        print("Invalid choice. Exiting.")
