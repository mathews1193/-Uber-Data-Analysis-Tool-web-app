from django.urls import path
from hello import views

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("aboutus/", views.aboutus, name="aboutus"),
]