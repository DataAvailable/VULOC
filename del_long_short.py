import os

def process_test():
    with open('test.txt', 'r') as f:
        lines = f.readlines()
        f.close()

    label_index = []
    function_list = []
    for i in range(len(lines)):
        if lines[i].startswith('__label'):
            label_index.append(i)
    for i in range(len(label_index)):
        if i < len(label_index) - 1:
            function_list.append(lines[label_index[i]: label_index[i + 1] - 1])
    print(function_list)
    f = open('./dataset/test.txt', 'a+')
    for i in range(len(function_list)):
        # if 10 < len(function_list[i]) < 80:
        #     for j in range(len(function_list[i])):
        #         f.write(function_list[i][j])
        #     f.write('\n')
        if 10 < len(function_list[i]) < 120:
            for j in range(len(function_list[i])):
                f.write(function_list[i][j])
            f.write('\n')
    f.close()
    os.system('rm test.txt')


def process_train():
    with open('train.txt', 'r') as f:
        lines = f.readlines()
        f.close()

    label_index = []
    function_list = []
    for i in range(len(lines)):
        if lines[i].startswith('__label'):
            label_index.append(i)
    for i in range(len(label_index)):
        if i < len(label_index) - 1:
            function_list.append(lines[label_index[i]: label_index[i + 1] - 1])
    print(function_list)
    f = open('./dataset/train.txt', 'a+')
    for i in range(len(function_list)):
        # if 10 < len(function_list[i]) < 80:
        #     for j in range(len(function_list[i])):
        #         f.write(function_list[i][j])
        #     f.write('\n')
        if 10 < len(function_list[i]) < 120:
            for j in range(len(function_list[i])):
                f.write(function_list[i][j])
            f.write('\n')
    f.close()
    os.system('rm train.txt')


process_train()
process_test()
