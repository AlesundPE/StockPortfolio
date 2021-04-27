from django.urls import path
from . import views

urlpatterns = [
    path('customers', views.listCustomers, name = "customer"),
    path('', views.home, name = "home"),
    path('about', views.about, name = "about"),
    path('addStock', views.addStock, name = "addStock"),
    path('delete/<stock_id>', views.delete, name="delete"),
    path('index', views.tvChart, name="index")
]