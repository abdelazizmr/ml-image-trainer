from django.shortcuts import render

# Create your views here.

# classifier/views.py
import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .ml_utils import train_model, predict_image

def index(request):
    """View for the first page - dataset creation and training"""
    return render(request, 'index.html')

def train(request):
    """Handle the training process"""
    if request.method == 'POST':
        # Process cat images
        for key, file in request.FILES.items():
            if key.startswith('cat_image_'):
                fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'cats'))
                fs.save(file.name, file)
        
        # Process dog images
        for key, file in request.FILES.items():
            if key.startswith('dog_image_'):
                fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'dogs'))
                fs.save(file.name, file)
        
        # Train the model
        accuracy = train_model(
            os.path.join(settings.MEDIA_ROOT, 'cats'),
            os.path.join(settings.MEDIA_ROOT, 'dogs')
        )
        
        # Redirect to prediction page with accuracy
        return redirect(f'/predict/?accuracy={accuracy:.2f}')
    
    return redirect('index')

def predict(request):
    """View for the second page - prediction"""
    accuracy = request.GET.get('accuracy', None)
    prediction = None
    confidence = None
    img_url = None
    
    if request.method == 'POST' and 'test_image' in request.FILES:
        # Save the uploaded image
        test_image = request.FILES['test_image']
        fs = FileSystemStorage()
        filename = fs.save(f"test_image.{test_image.name.split('.')[-1]}", test_image)
        img_url = fs.url(filename)
        
        # Make prediction
        model_path = os.path.join(settings.MEDIA_ROOT, 'model.pkl')
        if os.path.exists(model_path):
            prediction, confidence = predict_image(
                os.path.join(settings.MEDIA_ROOT, filename),
                model_path
            )
    
    return render(request, 'predict.html', {
        'accuracy': accuracy,
        'prediction': prediction,
        'confidence': confidence,
        'img_url': img_url
    })