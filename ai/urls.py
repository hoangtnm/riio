from django.urls import path
from . import views

app_name = 'ai'
urlpatterns = [
    path('', views.home, name="homepage"),
    path('upload/', views.upload, name='upload_file'),
    path('history/', views.history, name='display_history')
]
