import os
import cv2
import numpy as np
from abc import ABC, abstractmethod

class BasePestDetector(ABC):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def predict(self, image_path):
        pass
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Common preprocessing method for all detectors"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img
    
    def get_metrics(self, prediction_score):
        """Calculate quality, paddy_safety, and accuracy from prediction score"""
        # Convert prediction score (0-1) to quality (1-5)
        # For pest detection: higher score means higher chance of pest
        has_pest = prediction_score > 0.5
        
        if has_pest:
            # If pest detected, higher score means lower quality
            quality = max(1, 5 - int(prediction_score * 4))
            paddy_safety = 100 - (prediction_score * 100)
        else:
            # If no pest detected, higher score means higher quality
            quality = max(1, int((1 - prediction_score) * 5))
            paddy_safety = (1 - prediction_score) * 100
            
        # Accuracy is confidence in the prediction
        accuracy = max(prediction_score, 1 - prediction_score) * 100
        
        return {
            'has_pest': has_pest,
            'quality': quality,
            'paddy_safety': round(paddy_safety, 1),
            'accuracy': round(accuracy, 1)
        }