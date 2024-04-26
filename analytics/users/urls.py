from django.urls import path
from . import views

app_name = "users"

urlpatterns = [
    path('login/', views.login_user, name='login'), # http://127.0.0.1:8000/users:login
    path('logout/', views.logout_user, name='logout'), # http://127.0.0.1:8000/users:logout
]

