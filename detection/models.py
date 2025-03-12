from django.db import models
from django.contrib.auth.models import User
import os

def image_upload_path(instance, filename):
    return os.path.join('uploads', filename)

class PaddyImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to=image_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Image by {self.user.username} at {self.uploaded_at}"

class DetectionResult(models.Model):
    paddy_image = models.ForeignKey(PaddyImage, on_delete=models.CASCADE, related_name='results')
    algorithm_name = models.CharField(max_length=100)
    has_pest = models.BooleanField()
    quality = models.IntegerField()  # Quality score (1-5)
    paddy_safety = models.FloatField()  # Percentage safety
    accuracy = models.FloatField()  # Percentage accuracy
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Result for {self.paddy_image} using {self.algorithm_name}"