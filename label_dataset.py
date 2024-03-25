import re
import random


def labeling():
    index = 0
    good_num = 0  # 文件个数
    bad_num = 0   # 坏文件个数
    rand = random.randrange(0, 2)
    f1 = open('./assemble/bad_function_assembly.txt', 'r')
    lines = f1.readlines()
    file_index = []
    vulner_list = []
    vulner_funcname_list = []
    for i in range(len(lines)):
        if lines[i].startswith('----------------------------------'):
            file_index.append(i)
            vulner_list.append(lines[i - 2].strip())
            vulner_funcname_list.append(lines[i-1])     # 存放源文件路径

    file_list = []
    for i in range(len(file_index)):
        if i + 1 < len(file_index):
            file_list.append(lines[file_index[i] + 1: file_index[i + 1] - 4])
    file_list.append(lines[file_index[-1] + 1: -2])

    f2 = open('./assemble/good_function_assembly.txt', 'r')
    lines_good = f2.readlines()
    file_index_good = []
    good_funcname_list = []
    for i in range(len(lines_good)):
        if lines_good[i].startswith('----------------------------------'):
            file_index_good.append(i)
            good_funcname_list.append(lines_good[i-1])       # 存放源文件路径
    file_list_good = []
    for i in range(len(file_index_good)):
        if i + 1 < len(file_index_good):
            file_list_good.append(lines_good[file_index_good[i] + 1: file_index_good[i + 1] - 4])
    file_list_good.append(lines_good[file_index_good[-1] + 1: -2])

    bad_length = len(file_list)
    good_length = len(file_list_good)
    # bad
    # for f in range(len(file_list)):
    #     fuction_index_bad = []
    #     print(file_list[f])
    #     for i in range(len(file_list[f])):
    #         if file_list[f][i].startswith('Dump'):
    #             fuction_index_bad.append(i)
    #     fuction_list_bad = []
    #     for i in range(len(fuction_index_bad)):
    #         if i + 1 < len(fuction_index_bad):
    #             fuction_list_bad.append(file_list[f][fuction_index_bad[i] + 1: fuction_index_bad[i + 1]])
    #             fuction_list_bad[i].insert(0, vulner_funcname_list[f])      # 在每个函数的第一行加入源文件名
    #     fuction_list_bad.append(file_list[f][fuction_index_bad[-1] + 1:])
    #     fuction_list_bad[-1].insert(0, vulner_funcname_list[f])            # 在每个函数的第一行加入源文件名(最后一个函数)
    #     for function in fuction_list_bad:
    #         # print(function)
    #         label_function(function, vulner_list[index], bad_num, bad_length, good_num, good_length, rand)
    #     bad_num += 1
    #     index += 1
    # f1.close()

    for f in range(len(file_list_good)):
        # bad
        # if rand == 0:
        fuction_index_bad = []
        print(file_list[f])
        for i in range(len(file_list[f])):
            if file_list[f][i].startswith('Dump'):
                fuction_index_bad.append(i)
        fuction_list_bad = []
        for i in range(len(fuction_index_bad)):
            if i + 1 < len(fuction_index_bad):
                fuction_list_bad.append(file_list[f][fuction_index_bad[i] + 1: fuction_index_bad[i + 1]])
                fuction_list_bad[i].insert(0, vulner_funcname_list[f])  # 在每个函数的第一行加入源文件名
        fuction_list_bad.append(file_list[f][fuction_index_bad[-1] + 1:])
        fuction_list_bad[-1].insert(0, vulner_funcname_list[f])  # 在每个函数的第一行加入源文件名(最后一个函数)
        for function in fuction_list_bad:
            # print(function)
            label_function(function, vulner_list[index], bad_num, bad_length, good_num, good_length, rand)
        bad_num += 1
        index += 1
        # else:
            # good
        fuction_index = []
        print(file_list_good[f])
        for i in range(len(file_list_good[f])):
            if file_list_good[f][i].startswith('Dump'):
                fuction_index.append(i)
        fuction_list = []
        for i in range(len(fuction_index)):
            if i + 1 < len(fuction_index):
                fuction_list.append(file_list_good[f][fuction_index[i] + 1: fuction_index[i + 1]])
                fuction_list[i].insert(0, good_funcname_list[f])        # 在每个函数的第一行加入源文件名
        fuction_list.append(file_list_good[f][fuction_index[-1] + 1:])
        fuction_list[-1].insert(0, good_funcname_list[f])              # 在每个函数的第一行加入源文件名(最后一个函数)
        for function in fuction_list:
            # print(function)
            write_dataset('good', function, bad_num, bad_length, good_num, good_length, rand)
        good_num += 1
    f1.close()
    f2.close()


# def labeling_good():
    # f2 = open('./assemble/good_function_assembly.txt', 'r')
    # lines_good = f2.readlines()
    # file_index_good = []
    # for i in range(len(lines_good)):
    #     if lines_good[i].startswith('----------------------------------'):
    #         file_index_good.append(i)
    # file_list_good = []
    # for i in range(len(file_index_good)):
    #     if i + 1 < len(file_index_good):
    #         file_list_good.append(lines_good[file_index_good[i] + 1: file_index_good[i + 1] - 4])
    # file_list_good.append(lines_good[file_index_good[-1] + 1: -2])
    # for file in file_list_good:
    #     fuction_index = []
    #     print(file)
    #     for i in range(len(file)):
    #         if file[i].startswith('Dump'):
    #             fuction_index.append(i)
    #     fuction_list = []
    #     for i in range(len(fuction_index)):
    #         if i + 1 < len(fuction_index):
    #             fuction_list.append(file[fuction_index[i] + 1: fuction_index[i + 1]])
    #     fuction_list.append(file[fuction_index[-1] + 1:])
    #     for function in fuction_list:
    #         # print(function)
    #         write_dataset('good', function, num)
    #     num += 1


def label_function(func_list, vulner_line, bad_num, bad_length, good_num, good_length, rand):
    vulnerability = False
    for line in func_list:
        if line != '\n' and line != '':
            if line.split(' ')[0] == vulner_line:
                vulnerability = True
    if vulnerability:
        write_dataset('bad', func_list, bad_num, bad_length, good_num, good_length, rand)
    else:
        write_dataset('good', func_list, bad_num, bad_length, good_num, good_length, rand)


def write_dataset(func_type, func_list, bad_num, bad_length, good_num, good_length, rand):
    # if total_file_num * 0.8 < num:
    #     f = open('./dataset/train.txt', 'a+')
    # else:
    #     f = open('./dataset/test.txt', 'a+')
    # if 0 < num - int(total_file_num * 0.05) < int(total_file_num * 0.25):
    if int(bad_length * 0.8) > bad_num or (bad_length == bad_num and int(good_length * 0.8) > good_num):
        f = open('./dataset/train.txt', 'a+')
        if func_type == 'good':
            f.write('__label0__\n')

            for line in func_list:
                line = re.sub(r'(\s*\,\s*)', ' , ', line)
                line = line.replace('(', ' ( ')
                line = line.replace(')', ' ) ')
                f.write(line)
            f.write('\n')
        elif func_type == 'bad':
            f.write('__label1__\n')
            for line in func_list:
                line = re.sub(r'(\s*\,\s*)', ' , ', line)
                line = line.replace('(', ' ( ')
                line = line.replace(')', ' ) ')
                f.write(line)
            f.write('\n')
        else:
            print('error!')
        f.close()
    # elif num < int(total_file_num * 0.05):
    else:
        f = open('./dataset/test.txt', 'a+')
        if func_type == 'good':
            f.write('__label0__\n')
            for line in func_list:
                line = re.sub(r'(\s*\,\s*)', ' , ', line)
                line = line.replace('(', ' ( ')
                line = line.replace(')', ' ) ')
                f.write(line)
            f.write('\n')
        elif func_type == 'bad':
            f.write('__label1__\n')
            for line in func_list:
                line = re.sub(r'(\s*\,\s*)', ' , ', line)
                line = line.replace('(', ' ( ')
                line = line.replace(')', ' ) ')
                f.write(line)
            f.write('\n')
        else:
            print('error!')
    # if func_type == 'good':
    #     f.write('__label1__\n')
    #     for line in func_list:
    #         f.write(line)
    #     f.write('\n')
    # elif func_type == 'bad':
    #     f.write('__label0__\n')
    #     for line in func_list:
    #         f.write(line)
    #     f.write('\n')
    # else:
    #     print('error!')
        f.close()


labeling()
