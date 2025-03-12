from .random_forest import RandomForestDetector
from .cnn import CNNDetector
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from .base_detector import BasePestDetector

class MobileNetDetector(BasePestDetector):
    def __init__(self, model_path=None):
        super().__init__(model_path)
        self.name = "MobileNet"
        self.target_size = (224, 224)
        
        # Check if model exists
        if model_path and os.path.exists(model_path):
            self.load_model()
            self.is_trained = True
        else:
            # Create a transfer learning model
            self.create_model()
            self.is_trained = False
    
    def create_model(self):
        """Create a transfer learning model using MobileNetV2"""
        # Load the base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(224, 224, 3))
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
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
    
    def preprocess_for_mobilenet(self, image_path):
        """Preprocess image for MobileNet input"""
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 127.5 - 1  # Normalize to [-1, 1]
        return img_array
    
    def predict(self, image_path):
        """Predict if the image has pests"""
        img_array = self.preprocess_for_mobilenet(image_path)
        
        # If model isn't trained, return a random prediction for demo purposes
        if not self.is_trained:
            prediction_score = np.random.uniform(0.4, 0.6)
        else:
            # Get prediction probability
            prediction = self.model.predict(img_array)
            prediction_score = prediction[0][0]
        
        # Get metrics based on prediction score
        return self.get_metrics(prediction_score)

# Define which algorithms are available
AVAILABLE_ALGORITHMS = {
    'random_forest': RandomForestDetector,
    'cnn': CNNDetector,
    'mobilenet': MobileNetDetector
}