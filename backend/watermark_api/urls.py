from django.urls import path
from . import views

urlpatterns = [
    path('test/', views.test_connection, name='test_connection'),
    path('protect/', views.protect_audio, name='protect_audio'),
    path('check/', views.check_audio, name='check_audio'), 
    path('benchmark/', views.run_benchmark, name='run_benchmark'),
]