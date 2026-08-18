from django.urls import path
from . import views

urlpatterns = [
    path("", views.weekly_report, name="home"),
    path("weekly-report/", views.weekly_report, name="weekly-report"),
]