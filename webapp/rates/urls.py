from django.urls import path

from . import views

urlpatterns = [
    path('treasuries', views.treasuries, name='treasuries'),


]