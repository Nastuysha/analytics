from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import MinLengthValidator, MaxLengthValidator
from django.utils.deconstruct import deconstructible

from head.models import UploadFiles, ModelML, ModelTS


class UploadFileForm(forms.Form):
    file = forms.FileField()


class DynamicChoiceForm(forms.Form):
    axis_choice = forms.ChoiceField(choices=[])

    def __init__(self, *args, **kwargs):
        dynamic_choices = kwargs.pop('dynamic_choices', [])
        super(DynamicChoiceForm, self).__init__(*args, **kwargs)
        self.fields['axis_choice'].choices = dynamic_choices


class SecondForm(forms.Form):
    SECOND_FORM_CHOICES = [
        ('linear_regression', 'Linear Regression'),
        ('knn', 'KNN'),
        ('decision_tree', 'Decision Tree')
    ]
    second_choice = forms.ChoiceField(choices=SECOND_FORM_CHOICES)


class MultiFormPredict(forms.Form):
    axis_choice = forms.ChoiceField(choices=[])
    second_choice = forms.ModelChoiceField(queryset=ModelML.objects.all())

    def __init__(self, *args, **kwargs):
        dynamic_choices = kwargs.pop('dynamic_choices', [])
        super(MultiFormPredict, self).__init__(*args, **kwargs)
        self.fields['axis_choice'].choices = dynamic_choices


class MultiFormSeries(forms.Form):
    axis_choice = forms.ChoiceField(choices=[])
    second_choice = forms.ModelChoiceField(queryset=ModelTS.objects.all())

    def __init__(self, *args, **kwargs):
        dynamic_choices = kwargs.pop('dynamic_choices', [])
        super(MultiFormSeries, self).__init__(*args, **kwargs)
        self.fields['axis_choice'].choices = dynamic_choices
