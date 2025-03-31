import os
from os import path


# 扫描测试用例的所有文件
def scaner_file(url, save_base_path):
    relative_url = url
    if path.isfile(relative_url):  # 判断是否为文件
        if relative_url.endswith('.c'):  # 判断文件名后缀是否为c
            compile_file(relative_url)
    elif path.isdir(relative_url):  # 判断是否为文件夹
        scaner_file(relative_url, save_base_path)
    else:
        print("error")
        pass
    # return good_output, bad_output
    # return output


def compile_file(filepath):  # 编译文件
    header_path = ' -I ./testcasesupport/ '  # 设置头文件路径
    run_main = ' -D INCLUDEMAIN '  # 编译时加入宏定义, 即单个测试样例编译时运行main函数

    good_func = ' -D OMITBAD '  # 编译时过滤坏(bad)函数
    bad_func = ' -D OMITGOOD '  # 编译时过滤好(good)函数

    filename = get_filename(filepath)

    good_filename = filename + '_good'
    bad_filename = filename + '_bad'

    good_save_path = './BinaryFile/good/'  # 好文件存放路径
    bad_save_path = './BinaryFile/bad/'  # 坏文件存放路径

    if not os.path.exists(good_save_path):
        os.system('mkdir ' + good_save_path)
    if not os.path.exists(bad_save_path):
        os.system('mkdir ' + bad_save_path)

    # good_output = subprocess.Popen('gcc -g -o ' + good_save_path + good_filename + header_path + run_main +
    # good_func + filepath, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')  #
    # 编译文件里的好函数,得到好函数的二进制文件
    os.system('gcc -g -o ' + good_save_path + good_filename + header_path + run_main + good_func + filepath)
    os.system('gcc -g -o ' + bad_save_path + bad_filename + header_path + run_main + bad_func + filepath)

    # bad_output = subprocess.Popen('gcc -g -o ' + bad_save_path + bad_filename + header_path + run_main + bad_func +
    # filepath, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')  # 编译文件里的坏函数,
    # 得到坏函数的二进制文件
    # return good_output, bad_output

    print('Saved in ' + good_save_path + good_filename + '\n' 'Saved in ' + bad_save_path \
          + bad_filename)
    # return output


def get_filename(filepath):  # 获取文件名
    sourcefile = filepath.split('/')[-1]
    filename, filetype = sourcefile.split('.')
    return filename

# sourcecode_path = './testcases/'
# save_base_dir = './BinaryFile/'
# scaner_file(sourcecode_path, save_base_dir)
