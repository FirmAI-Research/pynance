from django.urls import path

from . import views

urlpatterns = [
    path('fundamentals', views.fundamentals, name='fundamentals'),
    path('sector_performance', views.sector_performance, name='sector_performance'),


]