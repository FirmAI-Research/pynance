from django.urls import path

from . import views

urlpatterns = [
    path('models', views.time_series_models, name='time_series_models'),
]