import os
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from .models import PaddyImage, DetectionResult
from .forms import PaddyImageUploadForm
from .algorithms import AVAILABLE_ALGORITHMS

def home(request):
    return render(request, 'detection/home.html')

@login_required
def upload_image(request):
    if request.method == 'POST':
        form = PaddyImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the image
            paddy_image = form.save(commit=False)
            paddy_image.user = request.user
            paddy_image.save()
            
            # Process the image with algorithms
            process_image(paddy_image)
            
            return redirect('results', image_id=paddy_image.id)
    else:
        form = PaddyImageUploadForm()
    
    return render(request, 'detection/upload.html', {'form': form})

def process_image(paddy_image):
    """Process the image with all available algorithms"""
    # Get the image path
    image_path = os.path.join(settings.MEDIA_ROOT, paddy_image.image.name)
    
    # Process with each algorithm
    for algo_key, algo_class in AVAILABLE_ALGORITHMS.items():
        # Initialize detector
        detector = algo_class()
        
        # Make prediction
        results = detector.predict(image_path)
        
        # Save results to database
        DetectionResult.objects.create(
            paddy_image=paddy_image,
            algorithm_name=detector.name,
            has_pest=results['has_pest'],
            quality=results['quality'],
            paddy_safety=results['paddy_safety'],
            accuracy=results['accuracy']
        )

@login_required
def results(request, image_id):
    try:
        paddy_image = PaddyImage.objects.get(id=image_id, user=request.user)
        results = DetectionResult.objects.filter(paddy_image=paddy_image)
        
        context = {
            'paddy_image': paddy_image,
            'results': results
        }
        
        return render(request, 'detection/results.html', context)
    except PaddyImage.DoesNotExist:
        messages.error(request, 'Image not found or you do not have permission to view it.')
        return redirect('home')