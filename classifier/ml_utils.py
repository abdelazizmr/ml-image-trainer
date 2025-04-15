# classifier/ml_utils.py
import os
import numpy as np
from sklearn.svm import SVC
from PIL import Image
import pickle

def extract_features(image_path):
    """Extract simple features from an image (resize and flatten)"""
    try:
        img = Image.open(image_path).resize((50, 50)).convert('L')  # resize to 50x50 and convert to grayscale
        return np.array(img).flatten() / 255.0  # normalize pixel values
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def train_model(cat_dir, dog_dir):
    """Train a simple SVM model on cat and dog images"""
    features = []
    labels = []

    # Process cat images
    for img_file in os.listdir(cat_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(cat_dir, img_file)
            img_features = extract_features(img_path)
            if img_features is not None:
                features.append(img_features)
                labels.append('cat')
    
    # Process dog images
    for img_file in os.listdir(dog_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dog_dir, img_file)
            img_features = extract_features(img_path)
            if img_features is not None:
                features.append(img_features)
                labels.append('dog')
    
    # Train the model if we have data
    if len(features) > 0:
        # Create and train SVM model
        model = SVC(kernel='linear', probability=True)
        model.fit(features, labels)
        
        # Save the model
        model_path = os.path.join(os.path.dirname(cat_dir), 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Calculate basic accuracy (this is simplified)
        # In a real application, you would use cross-validation
        predictions = model.predict(features)
        accuracy = np.mean(predictions == labels) * 100
        return accuracy
    
    return 0

def predict_image(image_path, model_path):
    """Predict whether an image contains a cat or dog"""
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 0

    # Extract features
    features = extract_features(image_path)
    print(features[:10])
    if features is None:
        return None, 0

    try:
        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Get confidence percentage for the prediction
        class_index = 0 if prediction == 'cat' else 1
        confidence = probabilities[class_index] * 100

        if confidence < 50:
            return "uncertain", 0
        else:
            return prediction, confidence
    except Exception as e:
        print(f"Error predicting: {e}")
        return None, 0