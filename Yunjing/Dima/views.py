# from django.contrib.auth.models import User
import subprocess

from django.shortcuts import render
from django.contrib.auth.hashers import make_password, check_password
from django.http import HttpResponse, HttpResponseRedirect
# from models import User
# Create your views here.
from Dima.models import User
import json

from compile import *
from prediction import *
from slice_dataset import labeling, process_test


def is_login(func):
    def check_login(request):
        ticket = request.COOKIES.get('ticket')
        if not ticket:
            print("not ticket")
            return HttpResponseRedirect('../')
        user = User.objects.filter(ticket=ticket)
        if not user:
            print("user")
            return HttpResponseRedirect('../')
        return func(request)
    return check_login

def regist(request):  # 注册页面
    if request.method == 'GET':
        return render(request, 'regist.html')
    if request.method == 'POST':
        password = make_password(request.POST.get('password'))
        User.objects.create(u_name=request.POST.get('name'), u_password=password)
        print('注册成功')
        return HttpResponseRedirect('../login')


def login(request):  # 登陆页面
    if request.method == 'GET':
        return render(request, 'login.html')
    if request.method == 'POST':
        if User.objects.filter(u_name=request.POST.get('name')).exists():
            user = User.objects.filter(u_name=request.POST.get('name'))[0]
            if check_password(request.POST.get('password'), user.u_password):
                response = HttpResponseRedirect('../detect')
                response.set_cookie('ticket', 'test')
                user.ticket = 'test'
                user.save()
                print('登陆成功')
                return response
        return HttpResponse('登录失败')
    else:
        return HttpResponse('用户名错误')


def index(request):
    return render(request, "index.html")

def clearFile(request):
    if request.method == 'GET':
        os.system("rm -rf /home/lxh/Yunjing/assemble/*")
        os.system("rm -rf /home/lxh/Yunjing/dataset/*")
        os.system("rm -rf /home/lxh/Yunjing/BinaryFile/bad/*")
        os.system("rm -rf /home/lxh/Yunjing/BinaryFile/good/*")
        return HttpResponseRedirect('../detect')

@is_login
def detect(request):  # 登录成功后的页面
    if request.method == 'GET':
        return render(request, 'detect.html')
    if request.method == 'POST':
        # 获取上传的文件数量
        filenumber = request.POST.get('filenumber')
        # 定义全局文件名列表
        fileNameList = []
        binary_path = "./BinaryFile"
        if int(filenumber) > 1:
            # 获取前端页面输入的内容(单个为源代码，多个为代码名称)
            source_code = request.POST.get('sourcecode')
            filename_list = source_code.split(',')
            # 赋值给全局文件名列表，用于后续传递给result页面
            fileNameList = filename_list
            for filename in filename_list:
                source_path = "./Dima/testcases/" + filename
                # 通过GCC编译C程序为二进制(Binary)
                scaner_file(source_path, binary_path)
        else:
            # 获取上传的文件名称
            filename = request.POST.get('filename')
            # 赋值给全局文件名列表，用于后续传递给result页面
            fileNameList = filename
            # 写入到test.c文件中
            source_path = "./Dima/testcases/" + filename
            # 通过GCC编译C程序为二进制(Binary)
            scaner_file(source_path, binary_path)
        # 使用GDB反编译二进制为汇编语言，并利用addr2line将地址转化为行号
        out = subprocess.Popen('gdb -q -x disassemble_addr2line.py', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding='utf-8')
        output = out.communicate()[0]
        # 开始切片
        labeling()
        process_test()
        # 预测漏洞
        model = request.POST.get('model')
        output = deep_learning(model)
        with open('Dima/static/result.txt', 'w') as f:
            for out in output:
            	f.write(out)
        return render(request, "result.html", {'result': output, 'fileNameList':fileNameList})

@is_login
def result(request):  # 检测后返回结果的页面
    if request.method == 'GET':
        return render(request, 'result.html')
    if request.method == 'POST':
        return HttpResponseRedirect('../saveresult')

def logout(request):  # 注销登录
    if request.method == 'GET':
        # response = HttpResponse()
        response = HttpResponseRedirect('../')
        response.delete_cookie('ticket')
        print("注销成功")
        return response

def about(request):
    if request.method == 'GET':
        return render(request, 'about.html')

def speaker(request):
    if request.method == 'GET':
        return render(request, 'speaker.html')

def agenda(request):
    if request.method == 'GET':
        return render(request, 'agenda.html')

def venue(request):
    if request.method == 'GET':
        return render(request, 'venue.html')

def ticket(request):
    if request.method == 'GET':
        return render(request, 'ticket.html')
        
def vul_detail(request):
    if request.method == 'GET':
        return render(request, 'vul_detail.html')

def addstu(request):  # 注册成功允许跳转页面
    if request.method == 'GET':
        ticket = request.COOKIES.get('ticket')
        if not ticket:
            return HttpResponse('用户无法登录')
        user = User.objects.filter(ticket=ticket)
        if user:
            return render(request, 'demo.html')
        else:
            return HttpResponse("用户无法登录系统")

