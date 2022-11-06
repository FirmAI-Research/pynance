from django.urls import path

from . import views

urlpatterns = [
    path('fundamentals', views.fundamentals, name='fundamentals'),

]