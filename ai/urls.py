from django.urls import path
from . import views

app_name = 'ai'
urlpatterns = [
    path('', views.upload_file, name='upload_file'),
    path('history/', views.display_history, name='display_history')
]
