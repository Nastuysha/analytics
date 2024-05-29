from django.contrib.auth.views import LogoutView
from django.urls import path
from . import views

app_name = "users"

urlpatterns = [
    path('login/', views.LoginUser.as_view(), name='login'), # http://127.0.0.1:8000/users:login
    path('logout/', LogoutView.as_view(), name='logout'),# http://127.0.0.1:8000/users:logout
    path('register/', views.RegisterUser.as_view(), name='register'),
]

