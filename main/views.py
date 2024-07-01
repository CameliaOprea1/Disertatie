from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
from django.views.generic import ListView


def index(request):
    my_dict = {'insert_me':"Hello I am from views.py"}
    return render(request,'index.html', context=my_dict)

