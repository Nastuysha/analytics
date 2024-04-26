from django import template
from django.db.models import Count

import head.views as views
from head.models import Category

register = template.Library()

@register.simple_tag
def get_menu():
    return views.menu

@register.simple_tag()
def get_theory():
    return views.theory_analysis


@register.inclusion_tag('head/list_categories.html')
def show_categories(cat_selected=0):
    cats = Category.objects.annotate(total=Count("posts")).filter(total__gt=0)
    return {'cats': cats, 'cat_selected': cat_selected}

