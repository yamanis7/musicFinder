from django.contrib import admin
from django.urls import path, include
from musicapp import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("musicapp/", include("musicapp.urls")),
    path("", views.my_view),
]
