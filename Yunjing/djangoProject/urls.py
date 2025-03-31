"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.index, name='index')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='index')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import path,include
from django.urls import re_path as url
# from . import views,testdb,search
from Dima import views
urlpatterns = [
    path('', views.index),
    # url(r"login/^$", views.login),
    path('login/', views.login),
    url(r'^regist/$', views.regist),
    url(r'^detect/$', views.detect),
    path('getdata/', views.detect),     # 提交包含代码的表单，供后台检测
    path('saveresult/', views.result),
    path('logout/', views.logout),
    path('clearFile/', views.clearFile),
    url(r'^result/$', views.result),
    path('about/',views.about),
    path('speaker/',views.speaker),
    path('agenda/',views.agenda),
    path('venue/',views.venue),
    path('ticket/',views.ticket),
    path('vul_detail/',views.vul_detail),
]
