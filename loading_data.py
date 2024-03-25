from collections import Counter
from load_xml import read_xml
import nltk


# 加载数据集
def load_dataset():
    train_file = []
    test_file = []

    print('loading dataset...')
    # 读取所有训练集和测试集数据
    f = open('./dataset/train.txt', 'r')
    if f.mode == 'r':
        train_file = f.readlines()
        f.close()

    f = open('./dataset/test.txt', 'r')
    if f.mode == 'r':
        test_file = f.readlines()
        f.close()

    train_labels = []  # 提取并去除训练集中的标签
    main_indexes_train = []  # 存储"main"的列表：索引，也就是每个新的数据样本

    for i in range(len(train_file)):
        if train_file[i].startswith('__label0__'):
            train_labels.append(0)
            train_file[i] = 'main :\n'
            main_indexes_train.append(i)  # 添加main的索引到main_indexes_train列表中
        elif train_file[i].startswith('__label1__'):
            train_labels.append(1)
            train_file[i] = 'main :\n'
            main_indexes_train.append(i)

    test_labels = []  # 提取并去除测试集中的标签
    main_indexes_test = []  # 存储主要的列表：索引，也就是每个新的数据样本

    for i in range(len(test_file)):
        if test_file[i].startswith('__label0__'):
            test_labels.append(0)
            test_file[i] = 'main :\n'
            main_indexes_test.append(i)
        elif test_file[i].startswith('__label1__'):
            test_labels.append(1)
            test_file[i] = 'main :\n'
            main_indexes_test.append(i)

    # 标记化,创建字典，将所有单词映射到它在所有训练句子中出现的次数
    words = Counter()  # 计算某个值出现的次数
    for i, line in enumerate(train_file):  # train_file: 存放汇编代码的列表，列表中的每一个值代表汇编代码的每一行
        if not line.startswith('File'):         # 过滤文件路径
            token_list = nltk.word_tokenize(line)
            for word in token_list[2:]:  # nltk分词，例如将"mov eax , DWORD PTR [ ebp+12 ]" 分为 "mov" "eax" "," "DWORD"
                if not word.startswith("CWE"):
                    words.update([word])  # "PTR" "[" "ebp+12" "]"

    training_data = [0] * len(train_labels)  # 初始化一个长度等于训练样本个数(标签个数)的列表

    function_name_list = []  # 存放函数名
    # 存放每个样本中每一行汇编代码对应的源代码行号
    samples_line_list = [list() for i in range(len(training_data))]

    for x in range(len(training_data)):  # len(training_data)：训练样本个数（标签个数）
        # 获取每个样本函数所属源文件的文件名, 并存入列表中
        function_name_list.append(train_file[main_indexes_train[x]+1].split(' ')[1].split('/')[-1].split('.')[0])
        # 创建一个新的列表，其中存放单一的样本
        if x < (len(training_data) - 1):
            training_sample = [0] * (main_indexes_train[x + 1] - main_indexes_train[x] - 3)
        else:
            training_sample = [0] * (len(train_file) - main_indexes_train[x] - 3)  # 只有单一训练数据的情况

        # 遍历所有样本行，并将其添加到样本列表中
        # for j in range(len(training_sample)):  # len(training_sample): 每一个样本的长度
        if x < len(training_data) - 1:
            ind = 0
            for line in train_file[main_indexes_train[x]+2:main_indexes_train[x + 1]]:  # 提取训练数据(File与main之间的内容)
                if line != '\n':
                    # 训练样本: [['mov', 'eax', ',', 'DWORD', 'PTR', '[', 'ebp+12', ']'], [], ...]
                    training_sample[ind] = line.split()[2:]
                    samples_line_list[x].append(line.split()[0])        # 存放每个样本行对应的源代码行号
                ind += 1  # 列表中的每一个值代表训练样例中的每一行（分词过后的每一行汇编代码）
        else:
            ind = 0
            for line in train_file[main_indexes_train[x]+2:len(train_file)]:
                if line != '\n':
                    training_sample[ind] = line.split()[2:]
                ind += 1
        training_data[x] = training_sample  # training_data: 列表中的每一个值为一个经过分词后的训练样例

    test_data = [0] * len(test_labels)
    function_name_test_list = []        # 存放函数名
    test_line_list = [list() for i in range(len(test_data))]        # 存放每个样本中每一行汇编代码对应的源代码行号
    for x in range(len(test_data)):
        # 获取每个样本函数所属源文件的文件名, 并存入列表中
        function_name_test_list.append(test_file[main_indexes_test[x] + 1].split(' ')[1].split('/')[-1].split('.')[0])
        if x < (len(test_data) - 1):
            test_sample = [0] * (main_indexes_test[x + 1] - main_indexes_test[x] - 3)
        else:
            test_sample = [0] * (len(test_file) - main_indexes_test[x] - 3)

        for j in range(len(test_sample)):
            if x < len(test_data) - 1:
                ind = 0
                for line in test_file[main_indexes_test[x]+2:main_indexes_test[x + 1]]:
                    if line != '\n':
                        test_sample[ind] = line.split()[2:]
                        test_line_list[x].append(line.split()[0])       # 存放每个样本行对应的源代码行号
                    ind += 1
            else:
                ind = 0
                for line in test_file[main_indexes_test[x]+2:len(test_file)]:
                    if line != '\n':
                        test_sample[ind] = line.split()[2:]
                    ind += 1
        test_data[x] = test_sample

    # print(words)
    # print(len(words))
    # 获取每个文件内的漏洞行号
    line_dic = read_xml()

    del test_file, train_file
    return words, training_data, test_data, train_labels, test_labels, function_name_list, function_name_test_list, samples_line_list, test_line_list, line_dic


# load_dataset()
# def get_function_name():

