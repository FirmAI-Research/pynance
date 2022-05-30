from django.urls import path

from . import views

urlpatterns = [
    path('sector', views.fundamentals, name='fundamentals'),
    path('financials', views.financials, name='financials'),
    path('sec_reader', views.sec_reader, name='sec_reader'),

]
