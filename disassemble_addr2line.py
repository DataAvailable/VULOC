import os
import gdb
import re
from os import path
from xml.dom.minidom import parse
import xml.dom.minidom


def disassemble_binary(basepath, vulner_line_dic):
    files = os.listdir(basepath)
    if files:
        for file in files:
            filepath = path.join(basepath, file)
            if path.isfile(filepath):  # 判断路径是否为文件
                gdb.execute('file ' + filepath)
                if path.isfile(filepath):
                    functions = gdb.execute('info functions', to_string=True)  # 列出可执行文件中的所有函数名称
                    function_list = functions.split('\n')  # 存放所有函数名称的列表
                    debug_function_list = []  # 存放可调式函数名称的列表, 0:源文件路径 1:函数1 2:函数2 3:函数3...
                    for i in range(len(function_list)):
                        if function_list[i] == '' and i != 1:
                            debug_function_list = function_list[2:i]  # 取出可调试(自定义)函数名称
                            break
                    filename = ''
                    filename_path = ''
                    if debug_function_list:
                        filename_path = debug_function_list[0]
                        # debug_function_list[0] 可执行文件所属文件名
                        filename = filename_path.split()[1].split('/')[-1].split('.')[0]
                        vulner_line = vulner_line_dic[filename]  # 得到漏洞行号
                        if basepath == './BinaryFile/bad':
                            print(vulner_line)
                        print(filename_path)
                        print('----------------------------------')
                    else:
                        print('debug_function_list is NULL')
                    for function_statement in debug_function_list[1:]:
                        # 正则表达式, 匹配空字符/空格 与字符'('之间的内容. +?: 重复1次或更多次，但尽可能少重复
                        result_list = re.findall(r'\s(.+?)\(', function_statement)
                        function_name = result_list[0].split()[-1]  # 得到函数名
                        flows = gdb.execute('disassemble ' + function_name, to_string=True)  # 列出函数与汇编代码的映射
                        slice_sample2function(flows, filepath)
                    print('\n')
                else:
                    print('error!')
            elif path.isdir(filepath):
                disassemble_binary(filepath, vulner_line_dic)
            else:
                print('error!')


def slice_sample2function(flows, filepath):
    function_list = flows.split('\n')
    print(function_list[0])
    for line in function_list[1: len(function_list) - 3]:
        line = line.split()
        address = line[0]
        line[0] = address2line(address, filepath)   # 调用address2line
        line[1] = ':'
        print(line[0], line[1], ' '.join(str(x) for x in line[2: len(line)]))
    # print(flows.split('\n'))


def address2line(address, filepath):        # 通过addr2line将地址转化为代码行, 参数1：地址, 参数2：可执行文件
    r = os.popen("addr2line " + address + " -e " + filepath + ' -f -C -s')
    index = r.readlines()[1].strip().split(':')[1]      # 获取行数
    if len(index.split()) > 1:
        index = index.split()[0]
    return index


def read_xml():
    # 使用minidom解析器打开 XML 文档
    line_dic = {}
    DOMTree = xml.dom.minidom.parse("./manifest.xml")
    collection = DOMTree.documentElement
    if collection.hasAttribute("shelf"):
        print("Root element : %s" % collection.getAttribute("shelf"))
    # 在集合中获取所有testcase
    testcases = collection.getElementsByTagName("testcase")
    for testcase in testcases:  # 读取每一个testcase标签里的内容
        files = testcase.getElementsByTagName('file')
        for i in range(files.length):
            file = files[i]
            filepath = file.getAttribute("path")
            if filepath.endswith(".c"):
                name = filepath.split('.')[0]  # 去除文件名后缀
                # print(name)
                if file.getElementsByTagName("flaw"):  # 读取flaw标签
                    tag = file.getElementsByTagName("flaw")
                    for j in range(tag.length):
                        line = tag[j].getAttribute("line")  # 获取行号
                        line_dic[name] = line
                        # print(line)
    return line_dic  # 返回存放文件名以及相应的漏洞行数(或无漏洞)的字典


binary_path = './BinaryFile/good'  # or './BinaryFile/bad'
# gdb.execute("set logging redirect on")  # gdb输出在命令行不显示
if binary_path == './BinaryFile/good':
    gdb.execute("set logging on ./assemble/good_function_assembly.txt")  # 将输出存入文件
else:
    gdb.execute("set logging on ./assemble/bad_function_assembly.txt")
gdb.execute("set pagination off")  # 取消输出分页显示
line_dic = read_xml()
disassemble_binary(binary_path, line_dic)
