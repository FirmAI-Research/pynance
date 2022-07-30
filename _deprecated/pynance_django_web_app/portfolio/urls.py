from django.urls import path

from . import views

urlpatterns = [
    path('rebalance', views.rebalance, name='rebalance'),
    path('optimize', views.optimize, name='optimize'),
]
