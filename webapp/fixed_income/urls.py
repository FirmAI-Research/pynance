from django.urls import path

from . import views

urlpatterns = [
    path('treasuries', views.treasuries, name='treasuries'),
    path('inflation', views.inflation, name='inflation'),
    path('bonds', views.bonds, name='bonds'),


]