from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'), # http://127.0.0.1:8000/
    path('about/', views.about, name='about'), # http://127.0.0.1:8000/about/
    path('head/', views.head, name='head'), # http://127.0.0.1:8000/head/
    path('category/', views.categories, name='categories'), # http://127.0.0.1:8000/category
    path('category/<slug:cat_slug>/', views.show_category, name='cats_id'), # http://127.0.0.1:8000/category/1/
    path('registration/', views.registration, name='registration'), # http://127.0.0.1:8000/registration
    path('login/', views.authorization, name='login'), # http://127.0.0.1:8000/login
    path('<slug:url>/', views.start_page), # http://127.0.0.1:8000/1/
    path('theory/<slug:th_id>/', views.theory, name ='theory') # http://127.0.0.1:8000/theory/stats
]