from django.urls import path

from . import views

urlpatterns = [
    path('sector', views.fundamentals, name='fundamentals'),
    path('dcf', views.dcf, name='dcf'),

]
