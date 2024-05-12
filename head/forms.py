from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import MinLengthValidator, MaxLengthValidator
from django.utils.deconstruct import deconstructible

from head.models import UploadFiles

class UploadFileForm(forms.Form):
    file = forms.FileField()


