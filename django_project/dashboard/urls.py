from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="home"),
    path("weekly-report/", views.dashboard, name="weekly-report"),
]