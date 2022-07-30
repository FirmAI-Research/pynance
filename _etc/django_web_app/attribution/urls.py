from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='attribution'),
    path('famma_french', views.famma_french, name='famma_french'),
]
