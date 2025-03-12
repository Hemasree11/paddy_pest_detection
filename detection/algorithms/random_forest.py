import pickle
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from .base_detector import BasePestDetector

class RandomForestDetector(BasePestDetector):
    def __init__(self, model_path=None):
        super().__init__(model_path)
        self.name = "Random Forest"
        
        # If no model exists, we'll train a simple one
        if not model_path or not os.path.exists(model_path):
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.is_trained = False
        else:
            self.load_model()
            self.is_trained = True
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            return False
    
    def extract_features(self, image):
        """Extract basic features from the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Extract simple features
        avg_pixel = np.mean(gray)
        std_pixel = np.std(gray)
        
        # Simple edge detection for potential pest shapes
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        
        # Color distribution - pests might affect color
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        avg_hue = np.mean(hsv[:,:,0])
        avg_saturation = np.mean(hsv[:,:,1])
        
        # Combine all features
        features = np.array([
            avg_pixel, std_pixel, edge_density, 
            avg_hue, avg_saturation
        ]).reshape(1, -1)
        
        return features
    
    def train_demo_model(self, pest_dir, no_pest_dir):
        """Train a simple model for demonstration purposes"""
        features = []
        labels = []
        
        # Process pest images
        for img_name in os.listdir(pest_dir)[:20]:  # Limit to 20 for demo
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(pest_dir, img_name)
                img = self.preprocess_image(img_path)
                feat = self.extract_features(img)
                features.append(feat[0])
                labels.append(1)  # 1 = pest
        
        # Process no-pest images
        for img_name in os.listdir(no_pest_dir)[:20]:  # Limit to 20 for demo
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(no_pest_dir, img_name)
                img = self.preprocess_image(img_path)
                feat = self.extract_features(img)
                features.append(feat[0])
                labels.append(0)  # 0 = no pest
        
        # Train model if we have data
        if features and labels:
            self.model.fit(np.array(features), np.array(labels))
            self.is_trained = True
            return True
        
        return False
    
    def predict(self, image_path):
        """Predict if the image has pests"""
        img = self.preprocess_image(image_path)
        features = self.extract_features(img)
        
        # If model isn't trained, return a random prediction for demo purposes
        if not self.is_trained:
            prediction_score = np.random.uniform(0.4, 0.6)
        else:
            # Get prediction probability
            prediction = self.model.predict_proba(features)
            prediction_score = prediction[0][1]  # Probability of pest class
        
        # Get metrics based on prediction score
        return self.get_metrics(prediction_score)