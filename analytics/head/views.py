from django.http import HttpResponse, HttpResponseNotFound, Http404, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string

from head.forms import UploadFileForm

menu = [{'title': 'About site', 'content': 'about'},
        {'title': 'Registration', 'content': 'registration'},
        {'title': 'Authorization', 'content': 'login'}
        ]

type_of_analysis = [
    {'id': 1, 'title': 'stats', 'content': 'Output statistics'},
    {'id': 2, 'title': 'graphs', 'content': 'Output graphs'},
    {'id': 3, 'title': 'predict', 'content': 'Output predict'},
    {'id': 4, 'title': 'series', 'content': 'Output time series'},
]

theory_analysis = [
    {'id': 1, 'title': 'stats', 'content': 'Статистики'},
    {'id': 2, 'title': 'graphs', 'content': 'Графическое представление'},
    {'id': 3, 'title': 'predict', 'content': 'Машинное обучение'},
    {'id': 4, 'title': 'series', 'content': 'Сглаживание рядов'},
]


def index(request):  # HttpRequest
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(form.cleaned_data['file'])
    else:
        form = UploadFileForm()
    data = {
        'title': 'start page',
        'menu': menu,
        'form': form,
    }
    #t = render_to_string('head/index.html', context=data)  # Path to template: index.html
    return render(request, 'head/index.html', data)


def handle_uploaded_file(f):
    with open(f"uploads/{f.name}", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def about(request):  # HttpRequest
    t = render_to_string('head/about.html', {'title': 'About site', 'menu': menu})  # Path to template: about.html
    return HttpResponse(t)


def head(request):  # HttpRequest
    return HttpResponse('Head page app analytics')


def categories(request):  # HttpRequest
    data = {
        'title': 'categories of analysis',
        'type_of_analysis': type_of_analysis,
    }
    t = render_to_string('head/categories.html', context=data)  # Path to template: categories.html
    return HttpResponse(t)


def theory(request, th_id):  # HttpRequest
    if th_id != 'stats' and th_id != 'predict' and th_id != 'graphs' and th_id != 'series':
        raise Http404()
    return HttpResponse(f'<h1>Theory to categories of analysis</h1><p>id: {th_id}</p>')


def categories_id(request, cat_id):  # HttpRequest
    if cat_id != 'stats' and cat_id != 'predict' and cat_id != 'graphs' and cat_id != 'series':
        raise Http404()
    return HttpResponse(f'<h1>Categories of analysis</h1><p>id: {cat_id}</p>')


def start_page(request, url):  # HttpRequest
    return redirect('home')


def registration(request):  # HttpRequest
    return HttpResponse('Registration')


def authorization(request):  # HttpRequest
    return HttpResponse('login')


def page_not_found(request, exception):
    return HttpResponseNotFound('<h1>Page not found</h1>')
