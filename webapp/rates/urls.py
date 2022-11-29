from django.urls import path

from . import views

urlpatterns = [
    path('treasuries', views.treasuries, name='treasuries'),
    path('inflation', views.inflation, name='inflation'),
    path('fixed_income', views.fixed_income, name='fixed_income'),


]