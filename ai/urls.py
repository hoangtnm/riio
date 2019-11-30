from django.urls import path
from . import views

app_name = 'ai'
urlpatterns = [
    path('', views.upload, name='upload'),
    path('history/', views.history, name='history')
]
