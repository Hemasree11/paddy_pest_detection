from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload'),
    path('results/<int:image_id>/', views.results, name='results'),
]