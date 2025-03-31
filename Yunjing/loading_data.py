from collections import Counter
from xml.dom.minidom import parse
import xml.dom.minidom
import nltk
nltk.download('punkt')


# 加载数据集
def load_dataset():
    test_file = []

    print('读入数据...')
    # 读取所有数据

    f = open('./dataset/test.txt', 'r')
    if f.mode == 'r':
        test_file = f.readlines()
        f.close()

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
    for i, line in enumerate(test_file):  # train_file: 存放汇编代码的列表，列表中的每一个值代表汇编代码的每一行
        if not line.startswith('File'):  # 过滤文件路径
            token_list = nltk.word_tokenize(line)
            for word in token_list[2:]:  # nltk分词，例如将"mov eax , DWORD PTR [ ebp+12 ]" 分为 "mov" "eax" "," "DWORD"
                if not word.startswith("CWE"):
                    words.update([word])  # "PTR" "[" "ebp+12" "]"

    function_name_list = []  # 存放函数名
    test_data = [0] * len(test_labels)
    function_name_test_list = []  # 存放函数名
    test_line_list = [list() for i in range(len(test_data))]  # 存放每个样本中每一行汇编代码对应的源代码行号
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
                for line in test_file[main_indexes_test[x] + 2:main_indexes_test[x + 1]]:
                    if line != '\n':
                        test_sample[ind] = line.split()[2:]
                        test_line_list[x].append(line.split()[0])  # 存放每个样本行对应的源代码行号
                    ind += 1
            else:
                ind = 0
                for line in test_file[main_indexes_test[x] + 2:len(test_file)]:
                    if line != '\n':
                        test_sample[ind] = line.split()[2:]
                    ind += 1
        test_data[x] = test_sample

    # print(words)
    # print(len(words))
    # 获取每个文件内的漏洞行号
    line_dic = read_xml()

    del test_file
    return words, test_data, test_labels, function_name_list, function_name_test_list, test_line_list, line_dic


# 解析XML文档
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
# load_dataset()
# def get_function_name():
