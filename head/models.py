from django.conf import settings
from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
import os


# Create your models here.


class File(models.Model):
    id_account = models.IntegerField()
    file_name = models.CharField(max_length=255)
    time_create = models.DateTimeField(auto_now_add=True)
    time_update = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.file_name


class Category(models.Model):
    name = models.CharField(max_length=100, db_index=True, verbose_name="Category")
    slug = models.SlugField(max_length=255, unique=True, db_index=True)

    class Meta:
        verbose_name = "Category"
        verbose_name_plural = "Category"

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('category', kwargs={'cat_slug': self.slug})


# class UploadFiles(models.Model):
#     user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
#     file = models.FileField(upload_to='')
#
#     class Meta:
#          verbose_name = "upload files"
#          verbose_name_plural = "upload files"

def user_directory_path(instance, filename):
    # Формируем путь к папке пользователя внутри директории 'media'
    return f'{instance.user.username}/{filename}'


class UploadFiles(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    file = models.FileField(upload_to=user_directory_path)

    class Meta:
        verbose_name = "upload files"
        verbose_name_plural = "upload files"


class ModelML(models.Model):
    name = models.CharField(max_length=100, db_index=True, verbose_name="ModelML")

    class Meta:
        verbose_name = "ModelML"
        verbose_name_plural = "ModelML"

    def __str__(self):
        return self.name

