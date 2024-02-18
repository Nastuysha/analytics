from django.http import HttpResponse, HttpResponseNotFound, Http404
from django.shortcuts import render


def index(request):  # HttpRequest
    return HttpResponse('Page app analytics')


def head(request):  # HttpRequest
    return HttpResponse('Head page app analytics')


def categories(request):  # HttpRequest
    return HttpResponse('<h1>Categories of analysis</h1>')


def categories_id(request, cat_id):  # HttpRequest
    if cat_id != 'stats' and cat_id != 'vizual':
        raise Http404()
    return HttpResponse(f'<h1>Categories of analysis</h1><p>id: {cat_id}</p>')


def page_not_found(request, exception):
    return HttpResponseNotFound('<h1>Page not found</h1>')
