from django.http import HttpResponse, HttpResponseNotFound, Http404, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

from .forms import DynamicChoiceForm, MultiFormPredict, MultiFormSeries
import os
from django.conf import settings

from head.forms import UploadFileForm
from head.models import Category, UploadFiles

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.colors import ListedColormap

from django.contrib.auth.decorators import login_required
from django.core.files import File

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


def index(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Создаем объект модели UploadFiles на основе данных формы
            upload_file = UploadFiles(file=request.FILES['file'])

            # Проверяем, если пользователь авторизован
            if request.user.is_authenticated:
                upload_file.user = request.user

                # Сохраняем объект модели
                upload_file.save()
    else:
        form = UploadFileForm()
    data = {
        'title': 'start page',
        'menu': menu,
        'form': form,
    }
    return render(request, 'head/index.html', data)


# def index(request):  # HttpRequest
#     if request.method == "POST":
#         form = UploadFileForm(request.POST, request.FILES)
#         if form.is_valid():
#             #handle_uploaded_file(form.cleaned_data['file'])
#             fp = UploadFiles(file=form.cleaned_data['file'])
#             fp.save()
#     else:
#         form = UploadFileForm()
#     data = {
#         'title': 'start page',
#         'menu': menu,
#         'form': form,
#     }
#     #t = render_to_string('head/index.html', context=data)  # Path to template: index.html
#     return render(request, 'head/index.html', data)


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


# def show_category(request, cat_slug):  # HttpRequest
#     category = get_object_or_404(Category, slug=cat_slug)
#     #post = get_object_or_404(UploadFiles, id=4)
#     # if cat_slug != 'stats' and cat_slug != 'predict' and cat_slug != 'graphs' and cat_slug != 'series':
#     #    raise Http404()
#     data = {
#         'title': f'Categories of analysis: {category.name}',
#         'menu': menu,
#         #'post': post,
#         'cat_selected': cat_slug,
#     }
#     #print(post.file)
#     return render(request, 'head/post.html', data)
#     #return HttpResponse(f'<h1>Categories of analysis</h1><p>id: {cat_slug}</p>')


def start_page(request, url):  # HttpRequest
    return redirect('home')


def registration(request):  # HttpRequest
    return HttpResponse('Registration')


def authorization(request):  # HttpRequest
    return HttpResponse('login')


def page_not_found(request, exception):
    return HttpResponseNotFound('<h1>Page not found</h1>')


# def show_category(request, cat_slug):  # HttpRequest
#     category = get_object_or_404(Category, slug=cat_slug)
#     user_id = request.user.id
#     upload_file = get_object_or_404(UploadFiles, user_id=user_id)
#     file_path = upload_file.file.path
#
#     with open(file_path, 'rb') as file:
#         response = HttpResponse(file.read())
#         response['Content-Type'] = 'application/force-download'
#         response['Content-Disposition'] = 'attachment; filename=' + upload_file.file.name
#     data = {
#         'title': f'Categories of analysis: {category.name}',
#         'menu': menu,
#         'cat_selected': cat_slug,
#         'user_file': file,
#     }
#     return render(request, 'post.html', data)


def show_category(request, cat_slug):  # HttpRequest
    category = get_object_or_404(Category, slug=cat_slug)

    if request.user.is_authenticated:
        user_files = UploadFiles.objects.filter(user=request.user).order_by('-id')
        if user_files.exists():
            last_file = user_files.first()
            file_path = last_file.file.path
            file_name = last_file.file.name.split('/')[-1]  # Получаем имя файла

            if cat_slug == 'stats':
                mean_f, min_f, max_f, median_f = print_statistics(file_path)
                data = {
                    'title': f'Categories of analysis: {category.name}',
                    'menu': menu,
                    'stats_mean': mean_f,
                    'stats_min': min_f,
                    'stats_max': max_f,
                    'stats_median': median_f,
                }
                file = pd.read_csv(file_path, sep=',')
                return render(request, 'head/stats.html', data)
            elif cat_slug == 'graphs':
                df = pd.read_csv(file_path, sep=',')
                # Отображение первых пяти строк из CSV-файла
                # print(df.head())

                dynamic_choices = [(str(row_id), str(row_id)) for row_id in df.columns.tolist()]

                # Создаем экземпляр формы с динамическим списком значений
                form = DynamicChoiceForm(dynamic_choices=dynamic_choices)

                if request.method == 'POST':
                    form = DynamicChoiceForm(request.POST, dynamic_choices=dynamic_choices)
                    if form.is_valid():
                        selected_option = form.cleaned_data['axis_choice']
                        graphs_paths = print_graphic(file_path, str(selected_option))
                else:
                    graphs_paths = print_graphic(file_path, df[df.columns.tolist()[0]])
                media_graphs_paths = [os.path.join(settings.MEDIA_URL, os.path.relpath(path, settings.MEDIA_ROOT)) for
                                      path in graphs_paths]

                data = {
                    'title': f'Categories of analysis: {category.name}',
                    'menu': menu,
                    'graphs': media_graphs_paths,
                    'form': form,
                }

                return render(request, 'head/graphs.html', data)
            elif cat_slug == 'predict':
                df = pd.read_csv(file_path, sep=',')
                dynamic_choices_target = [(str(row_id), str(row_id)) for row_id in df.columns.tolist()]
                if request.method == 'POST':
                    # dynamic_form = DynamicChoiceForm(request.POST, dynamic_choices=dynamic_choices_target)
                    # second_form = SecondForm(request.POST)
                    form = MultiFormPredict(request.POST, dynamic_choices=dynamic_choices_target)

                    # print(dynamic_form.is_valid(), second_form.is_valid())
                    # if dynamic_form.is_valid() and second_form.is_valid():
                    if form.is_valid():
                        # selected_option = dynamic_form.cleaned_data['axis_choice']
                        # second_selected_option = second_form.cleaned_data['second_choice']
                        selected_option = form.cleaned_data['axis_choice']
                        second_selected_option = form.cleaned_data['second_choice'].name

                        print(selected_option, second_selected_option)
                        metric, predict_paths = print_ml(file_path, str(selected_option), str(second_selected_option))
                        # print('!!', predict_paths)
                        if predict_paths != 'err':
                            print(predict_paths)
                            media_predict_paths = os.path.join(settings.MEDIA_URL,
                                                               os.path.relpath(predict_paths, settings.MEDIA_ROOT))
                            data = {
                                'title': f'Categories of analysis: {category.name}',
                                'menu': menu,
                                'predict': media_predict_paths,
                                'form': form,
                                'metric': metric,
                                'test': selected_option,
                            }
                        else:
                            data = {
                                'title': f'Categories of analysis: {category.name}',
                                'menu': menu,
                                'predict': '',
                                'form': form,
                                'metric': metric,
                                'test': selected_option,
                            }
                        # print(data)
                        return render(request, 'head/predicts.html', data)
                else:
                    form = MultiFormPredict(dynamic_choices=dynamic_choices_target)

                    data = {
                        'title': f'Categories of analysis: {category.name}',
                        'menu': menu,
                        'predict': '',
                        'form': form,
                        'metric': -1,
                    }

                return render(request, 'head/predicts.html', data)
            elif cat_slug == 'series':
                df = pd.read_csv(file_path, sep=',')
                dynamic_choices_target = [(str(row_id), str(row_id)) for row_id in df.columns.tolist()]
                if request.method == 'POST':
                    # dynamic_form = DynamicChoiceForm(request.POST, dynamic_choices=dynamic_choices_target)
                    # second_form = SecondForm(request.POST)
                    form = MultiFormSeries(request.POST, dynamic_choices=dynamic_choices_target)

                    # print(dynamic_form.is_valid(), second_form.is_valid())
                    # if dynamic_form.is_valid() and second_form.is_valid():
                    if form.is_valid():
                        # selected_option = dynamic_form.cleaned_data['axis_choice']
                        # second_selected_option = second_form.cleaned_data['second_choice']
                        selected_option = form.cleaned_data['axis_choice']
                        second_selected_option = form.cleaned_data['second_choice'].name

                        print(selected_option, second_selected_option)
                        ts_paths = print_time_series(file_path, str(selected_option), str(second_selected_option))
                        # print('!!', predict_paths)
                        if ts_paths != 'err':
                            print(ts_paths)
                            media_ts_paths = os.path.join(settings.MEDIA_URL,
                                                          os.path.relpath(ts_paths, settings.MEDIA_ROOT))
                            data = {
                                'title': f'Categories of analysis: {category.name}',
                                'menu': menu,
                                'series': media_ts_paths,
                                'form': form,
                                'test': selected_option,
                            }
                        else:
                            data = {
                                'title': f'Categories of analysis: {category.name}',
                                'menu': menu,
                                'series': '',
                                'form': form,
                                'test': selected_option,
                            }
                        # print(data)
                        return render(request, 'head/series.html', data)
                else:
                    form = MultiFormSeries(dynamic_choices=dynamic_choices_target)

                    data = {
                        'title': f'Categories of analysis: {category.name}',
                        'menu': menu,
                        'series': '',
                        'form': form,
                    }

                return render(request, 'head/series.html', data)


def print_statistics(file_path):
    try:
        file = pd.read_csv(file_path, sep=',')

        features = file.columns.tolist()
        mean_features = []
        min_features = []
        max_features = []
        median_features = []

        for idx in range(len(features)):
            mean_features.append(str(features[idx]) + ": " + str(file[features[idx]].mean()))
            min_features.append(str(features[idx]) + ": " + str(file[features[idx]].min()))
            max_features.append(str(features[idx]) + ": " + str(file[features[idx]].max()))
            median_features.append(str(features[idx]) + ": " + str(file[features[idx]].median()))

        return mean_features, min_features, max_features, median_features
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to read file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def print_graphic(file_path, selected_option):
    file = pd.read_csv(file_path, sep=',')

    try:

        features = file.columns.tolist()

        name_x_feature = features[0]
        X = file[features[0]]
        # X_1 = file[str(selected_option)]
        # print(X_1)
        # if str(selected_option) == features[0]:
        for idx in range(len(features)):
            if file[features[idx]].dtype == 'float64' or file[features[idx]].dtype == 'int64':
                if str(selected_option) == features[idx]:
                    X = file[features[idx]]
                    name_x_feature = features[idx]
                    break
            # print(idx)
        # else:
        #    X = file[str(selected_option)]

        graphs_paths = []
        saved_path = ''
        directory_path = file_path[:file_path.rfind('/')]
        user_path = 'media' + file_path[directory_path.rfind('/'):file_path.rfind('/') + 1]

        # print(directory_path )
        for idx in range(len(features)):
            if file[features[idx]].dtype == 'float64' or file[features[idx]].dtype == 'int64':
                Y = file[features[idx]]

                plt.style.use('_mpl-gallery')

                # plot
                fig, ax = plt.subplots()
                ax.set_xlabel(name_x_feature, fontsize=26)
                ax.set_ylabel(features[idx], fontsize=26)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                ax.plot(X, Y, linewidth=2.0)
                fig.set_size_inches(10, 10)

                name_graphs = 'graphs_' + name_x_feature + '_' + features[idx] + '.png'
                saved_path = user_path + name_graphs.replace("/", "!")
                tmp_path = directory_path[:directory_path.rfind('/')]
                plt.savefig(saved_path, bbox_inches='tight')
                # print(saved_path)
                # tmp_path[:tmp_path.rfind('/')+1]
                graphs_paths.append(tmp_path[:tmp_path.rfind('/') + 1] + saved_path)
        return graphs_paths
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to read file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def print_ml(file_path, selected_option, second_selected_option):
    file = pd.read_csv(file_path, sep=',')
    features = file.columns.tolist()
    sel_option = features[0]
    sec_sel_option = 'decision_tree'
    for idx in range(len(features)):
        if file[features[idx]].dtype == 'float64' or file[features[idx]].dtype == 'int64':
            if str(selected_option) == features[idx]:
                sel_option = features[idx]
                sec_sel_option = second_selected_option
                break

    try:
        if sec_sel_option == 'linear_regression':
            metric, predict_paths = linear_regression(file, file_path, sel_option)
        elif sec_sel_option == 'knn':
            metric, predict_paths = knn(file, file_path, sel_option)
        else:
            metric, predict_paths = decision_tree(file, file_path, sel_option)
        print(metric, predict_paths)
        return metric, predict_paths
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to read file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def linear_regression(data, file_path, target):
    try:
        # разделение выборки
        data = data.dropna(subset=data.columns.tolist())
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)
        # работа с численными признаками
        numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

        X_train_num = X_train[numeric_features]
        X_test_num = X_test[numeric_features]

        scaler = StandardScaler()
        X_train_num_scaled = scaler.fit_transform(X_train_num)
        X_test_num_scaled = scaler.transform(X_test_num)
        # работа с категориальными признаками
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1)

        categorical_features = data.select_dtypes(include=['object']).columns.tolist()

        X_train_cat = encoder.fit_transform(X_train[categorical_features]).astype(int)
        X_test_cat = encoder.transform(X_test[categorical_features]).astype(int)
        # итоговые датасеты
        X_train = np.hstack((X_train_num_scaled, X_train_cat))
        X_test = np.hstack((X_test_num_scaled, X_test_cat))
        # обучение
        linear_regr = LinearRegression()
        linear_regr.fit(X_train, y_train)
        # предсказания + метрика
        pred = linear_regr.predict(X_test)
        metric = mean_absolute_percentage_error(y_test, pred)

        plt.plot(y_test, y_test)
        plt.scatter(pred, y_test, color='pink', s=30, edgecolor='black', linewidths=0.5)
        plt.xlabel('Предсказания')
        plt.ylabel('Тестовые данные')
        # plt.legend()
        plt.plot()

        directory_path = file_path[:file_path.rfind('/')]
        user_path = 'media' + file_path[directory_path.rfind('/'):file_path.rfind('/') + 1]

        name_graphs = 'pred_test.png'
        saved_path = user_path + name_graphs.replace("/", "!")
        tmp_path = directory_path[:directory_path.rfind('/')]

        plt.savefig(saved_path, bbox_inches='tight')

        graph_paths = (tmp_path[:tmp_path.rfind('/') + 1] + saved_path)
        return metric, graph_paths
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0, 'err'


def knn(data, file_path, target):
    try:
        # разделение выборки
        # data = data.dropna(subset=data.columns.tolist())
        # X = data.drop(columns=[target])
        # y = data[target]
        #
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)
        # # работа с численными признаками
        # numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
        #
        # X_train_num = X_train[numeric_features]
        # X_test_num = X_test[numeric_features]
        #
        # scaler = StandardScaler()
        # X_train_num_scaled = scaler.fit_transform(X_train_num)
        # X_test_num_scaled = scaler.transform(X_test_num)
        # # работа с категориальными признаками
        # encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1)
        #
        # categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        #
        # X_train_cat = encoder.fit_transform(X_train[categorical_features]).astype(int)
        # X_test_cat = encoder.transform(X_test[categorical_features]).astype(int)
        # # итоговые датасеты
        # X_train = np.hstack((X_train_num_scaled, X_train_cat))
        # X_test = np.hstack((X_test_num_scaled, X_test_cat))
        # # обучение
        # knn = KNeighborsClassifier()
        # knn.fit(X_train, y_train)
        # # предсказания + метрика
        # pred = knn.predict(X_test)
        # labelencoder = LabelEncoder()
        # y_test = labelencoder.fit_transform(y_test)
        # pred = labelencoder.transform(pred)
        # metric = f1_score(y_test, pred)
        #
        # plt.plot(y_test, y_test)
        # plt.scatter(pred, y_test, color='pink', s=30, edgecolor='black', linewidths=0.5)
        # plt.xlabel('Предсказания')
        # plt.ylabel('Тестовые данные')
        # # plt.legend()
        # plt.plot()
        data = data.dropna(subset=data.columns.tolist())
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)
        # работа с численными признаками
        numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
        numeric_features = numeric_features[:2]  # Ограничиваем до первых двух числовых признаков для визуализации

        X_train_num = X_train[numeric_features]
        X_test_num = X_test[numeric_features]

        scaler = StandardScaler()
        X_train_num_scaled = scaler.fit_transform(X_train_num)
        X_test_num_scaled = scaler.transform(X_test_num)

        # обучение
        knn = KNeighborsClassifier()
        knn.fit(X_train_num_scaled, y_train)

        # предсказания + метрика
        pred = knn.predict(X_test_num_scaled)
        labelencoder = LabelEncoder()
        y_test = labelencoder.fit_transform(y_test)
        pred = labelencoder.transform(pred)
        metric = f1_score(y_test, pred)

        # Визуализация результатов
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

        h = .02  # шаг сетки
        x_min, x_max = X_train_num_scaled[:, 0].min() - 1, X_train_num_scaled[:, 0].max() + 1
        y_min, y_max = X_train_num_scaled[:, 1].min() - 1, X_train_num_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Отобразим также обучающий набор
        plt.scatter(X_train_num_scaled[:, 0], X_train_num_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("K-Nearest Neighbors Classification")
        plt.plot()

        directory_path = file_path[:file_path.rfind('/')]
        user_path = 'media' + file_path[directory_path.rfind('/'):file_path.rfind('/') + 1]

        name_graphs = 'pred_test.png'
        saved_path = user_path + name_graphs.replace("/", "!")
        tmp_path = directory_path[:directory_path.rfind('/')]

        plt.savefig(saved_path, bbox_inches='tight')

        graph_paths = (tmp_path[:tmp_path.rfind('/') + 1] + saved_path)
        return metric, graph_paths
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0, 'err'


def decision_tree(data, file_path, target):
    try:
        # разделение выборки
        data = data.dropna(subset=data.columns.tolist())
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)
        # работа с численными признаками
        numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

        X_train_num = X_train[numeric_features]
        X_test_num = X_test[numeric_features]

        scaler = StandardScaler()
        X_train_num_scaled = scaler.fit_transform(X_train_num)
        X_test_num_scaled = scaler.transform(X_test_num)
        # работа с категориальными признаками
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1)

        categorical_features = data.select_dtypes(include=['object']).columns.tolist()

        X_train_cat = encoder.fit_transform(X_train[categorical_features]).astype(int)
        X_test_cat = encoder.transform(X_test[categorical_features]).astype(int)
        # итоговые датасеты
        X_train = np.hstack((X_train_num_scaled, X_train_cat))
        X_test = np.hstack((X_test_num_scaled, X_test_cat))
        # обучение
        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)
        # предсказания + метрика
        pred = tree.predict(X_test)
        labelencoder = LabelEncoder()
        y_test = labelencoder.fit_transform(y_test)
        pred = labelencoder.transform(pred)

        metric = f1_score(y_test, pred)

        # plt.plot(y_test, y_test)
        # plt.scatter(pred, y_test, color='pink', s=30, edgecolor='black', linewidths=0.5)
        # plt.xlabel('Предсказания')
        # plt.ylabel('Тестовые данные')
        # plt.legend()
        plt.figure(figsize=(20, 10))
        plot_tree(tree, filled=True, feature_names=X.columns.tolist(), class_names=np.unique(y).astype(str),
                  max_depth=3)
        plt.plot()

        directory_path = file_path[:file_path.rfind('/')]
        user_path = 'media' + file_path[directory_path.rfind('/'):file_path.rfind('/') + 1]

        name_graphs = 'pred_test.png'
        saved_path = user_path + name_graphs.replace("/", "!")
        tmp_path = directory_path[:directory_path.rfind('/')]

        plt.savefig(saved_path, bbox_inches='tight')

        graph_paths = (tmp_path[:tmp_path.rfind('/') + 1] + saved_path)
        return metric, graph_paths
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0, 'err'


def print_time_series(file_path, selected_option, second_selected_option):
    file = pd.read_csv(file_path, sep=',')
    features = file.columns.tolist()
    sel_option = features[0]
    sec_sel_option = 'SMA'
    for idx in range(len(features)):
        if file[features[idx]].dtype == 'float64' or file[features[idx]].dtype == 'int64':
            if str(selected_option) == features[idx]:
                sel_option = features[idx]
                sec_sel_option = second_selected_option
                break

    try:
        if sec_sel_option == 'SMA':
            ts_paths = time_series_SMA(file, file_path, sel_option)
        elif sec_sel_option == 'WMA':
            ts_paths = time_series_WMA(file, file_path, sel_option)
        else:
            ts_paths = time_series_EMA(file, file_path, sel_option)
        print(ts_paths)
        return ts_paths
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to read file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def sma(df, m):
    ans = []
    for i in range(len(df)):
        ans.append(np.mean([df.iloc[min(max(j, 0), min(j, len(df) - 1), key=lambda x: abs(len(df) // 2 - x))] for j in
                            range(i - m, i + m + 1)]))
    return np.array(ans)


def time_series_SMA(data, file_path, target):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(data[target], c='lightseagreen')
        plt.plot(pd.DataFrame(sma(data[target], 1)), c='mediumslateblue')
        plt.title(label='SMA')
        # plt.legend(['y', 'sma: w = 3'])
        plt.plot()

        directory_path = file_path[:file_path.rfind('/')]
        user_path = 'media' + file_path[directory_path.rfind('/'):file_path.rfind('/') + 1]

        name_graphs = 'series.png'
        saved_path = user_path + name_graphs.replace("/", "!")
        tmp_path = directory_path[:directory_path.rfind('/')]

        plt.savefig(saved_path, bbox_inches='tight')

        graph_paths = (tmp_path[:tmp_path.rfind('/') + 1] + saved_path)
        return graph_paths
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 'err'


# WMA
def wma(df, m):
    ans = []
    weights = np.array(
        [math.exp((-0.3) * abs(j)) / np.sum([math.exp((-0.3) * abs(i)) for i in range(-m, m + 1)]) for j in
         range(-m, m + 1)])
    for i in range(len(df)):
        ans.append(np.sum(np.array(
            [df.iloc[min(max(j, 0), min(j, len(df) - 1), key=lambda x: abs(len(df) // 2 - x))] for j in
             range(i - m, i + m + 1)]).T[0] * weights))
    return np.array(ans)


def time_series_WMA(data, file_path, target):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(data[target], c='lightseagreen')
        plt.plot(pd.DataFrame(wma(data[target], 1)), c='mediumslateblue')
        plt.title(label='WMA')
        # plt.legend(['y', 'wma: w = 3'])

        plt.plot()

        directory_path = file_path[:file_path.rfind('/')]
        user_path = 'media' + file_path[directory_path.rfind('/'):file_path.rfind('/') + 1]

        name_graphs = 'series.png'
        saved_path = user_path + name_graphs.replace("/", "!")
        tmp_path = directory_path[:directory_path.rfind('/')]

        plt.savefig(saved_path, bbox_inches='tight')

        graph_paths = (tmp_path[:tmp_path.rfind('/') + 1] + saved_path)
        return graph_paths
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 'err'


# EMA
def ema(df, alpha):
    ema_df = round(df['y'].ewm(alpha=alpha, adjust=False).mean(), 2)
    return ema_df


def time_series_EMA(data, file_path, target):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(data[target], c='lightseagreen')
        plt.plot(pd.DataFrame(ema(data[target], 0.7)), c='mediumslateblue')
        plt.title(label='EMA')
        # plt.legend(['y', 'ema: alpha = 0.7'])
        plt.plot()

        directory_path = file_path[:file_path.rfind('/')]
        user_path = 'media' + file_path[directory_path.rfind('/'):file_path.rfind('/') + 1]

        name_graphs = 'series.png'
        saved_path = user_path + name_graphs.replace("/", "!")
        tmp_path = directory_path[:directory_path.rfind('/')]

        plt.savefig(saved_path, bbox_inches='tight')

        graph_paths = (tmp_path[:tmp_path.rfind('/') + 1] + saved_path)
        return graph_paths
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 'err'
