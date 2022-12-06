from django.urls import path

from . import views

urlpatterns = [
    path('fundamentals', views.fundamentals, name='fundamentals'),
    path('sector_performance', views.sector_performance, name='sector_performance'),
    path('attribution', views.attribution, name='attribution'),
    path('etf_sector_performance', views.etf_sector_performance, name='etf_sector_performance'),


]