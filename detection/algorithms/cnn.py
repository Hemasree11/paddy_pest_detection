import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from .base_detector import BasePestDetector

class CNNDetector(BasePestDetector):
    def __init__(self, model_path=None):
        super().__init__(model_path)
        self.name = "CNN"
        self.target_size = (224, 224)
        
        # Check if model exists
        if model_path and os.path.exists(model_path):
            self.load_model()
            self.is_trained = True
        else:
            # Create a simple CNN model
            self.create_model()
            self.is_trained = False
    
    def create_model(self):
        """Create a simple CNN model for pest detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            self.model = load_model(self.model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_model()
            return False
    
    def preprocess_for_cnn(self, image_path):
        """Preprocess image for CNN input"""
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    
    def train_demo_model(self, pest_dir, no_pest_dir, epochs=5):
        """Train a simple model for demonstration purposes"""
        # This is simplified training just for demo
        X = []
        y = []
        
        # Process pest images
        for img_name in os.listdir(pest_dir)[:20]:  # Limit to 20 for demo
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(pest_dir, img_name)
                try:
                    img = load_img(img_path, target_size=self.target_size)
                    img_array = img_to_array(img) / 255.0
                    X.append(img_array)
                    y.append(1)  # 1 = pest
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Process no-pest images
        for img_name in os.listdir(no_pest_dir)[:20]:  # Limit to 20 for demo
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(no_pest_dir, img_name)
                try:
                    img = load_img(img_path, target_size=self.target_size)
                    img_array = img_to_array(img) / 255.0
                    X.append(img_array)
                    y.append(0)  # 0 = no pest
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Train model if we have data
        if X and y:
            X = np.array(X)
            y = np.array(y)
            self.model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=1)
            self.is_trained = True
            return True
        
        return False
    
    def predict(self, image_path):
        """Predict if the image has pests"""
        img_array = self.preprocess_for_cnn(image_path)
        
        # If model isn't trained, return a random prediction for demo purposes
        if not self.is_trained:
            prediction_score = np.random.uniform(0.3, 0.7)
        else:
            # Get prediction probability
            prediction = self.model.predict(img_array)
            prediction_score = prediction[0][0]
        
        # Get metrics based on prediction score
        return self.get_metrics(prediction_score)