from django.urls import path
from . import views

urlpatterns = [
    path('', views.index), # http://127.0.0.1:8000/
    path('head/', views.head), # http://127.0.0.1:8000/head/
    path('categories/', views.categories), # http://127.0.0.1:8000/categories/
    path('categories/<slug:cat_id>/', views.categories_id) # http://127.0.0.1:8000/categories/1/
]