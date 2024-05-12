"""analytics URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from analytics import settings
from head import views
from head.views import page_not_found

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('head.urls')),  # http://127.0.0.1:8000/
    path('users/', include('users.urls', namespace="users")),  # http://127.0.0.1:8000/
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

handler404 = page_not_found

admin.site.site_header = "Site administration"
admin.site.index_title = "File analytics"
