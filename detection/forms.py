from django import forms
from .models import PaddyImage

class PaddyImageUploadForm(forms.ModelForm):
    class Meta:
        model = PaddyImage
        fields = ['image']
        labels = {
            'image': 'Choose File',
        }