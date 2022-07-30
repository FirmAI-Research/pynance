from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='treasury_rates'),
    path('market_return_regression', views.market_return_regression, name='market_return_regression'),


]