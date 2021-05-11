from django.contrib import admin
from django.urls import path

from . import views

urlpatterns = [
    path('prediction/<symbol>', views.prediction, name = "prediction"),
]