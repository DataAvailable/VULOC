import numpy as np
from torch.utils.data import TensorDataset
import torch


# 寻找训练数据中最长代码行的函数
def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)


# 用于填充较短的代码行以匹配最长的代码行的函数。
def pad_seq_len(data, seq_len):
    features = np.zeros((len(data), seq_len), dtype=int)
    for ii, review in enumerate(data):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


# 用于填充较短样本的函数，以创建我们数据的统一三维矩阵
def pad_sample_len(data, longest_sample):
    diff = longest_sample - len(data)  # 最大样本与其它样本的长度差
    padding = [[0] * 2]  # 初始化一个2维矩阵
    for i in range(diff):
        data.extend(padding)
    return data


# 数据预处理
def data_preprocess(test_data):
    longest_sample_index = 0
    # 在测试数据中找到最长的样本：
    longest_sample_test = len(test_data[0])
    longest_sample_index_test = 0

    for i in range(len(test_data)):
        tmp = len(test_data[i])
        if tmp > longest_sample_test:
            longest_sample_test = tmp
            longest_sample_index_test = i

    longest_sample = longest_sample_test
    longest_sample_index = longest_sample_index_test
    # max_seq = find_max_list(test_data[longest_sample_index])
    max_seq = 10

    for i in range(len(test_data)):
        test_data[i] = pad_sample_len(test_data[i], longest_sample)
        test_data[i] = pad_seq_len(test_data[i], max_seq)

    return test_data, max_seq, longest_sample


# 加载数据
def covert_data_to_tensor(test_data, test_labels):
    # 将数据和标签转换为 numpy 数组
    test_labels = np.array(test_labels)
    test_data = np.asarray(test_data)
    # 为训练、验证和测试创建 tensorDatasets
    test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
    return test_data


# 生成01定位矩阵
def Generate_01_matrix(function_name_list, samples_line_list, line_dic, labels, samples_size, batch_size, index):
    # 创建长度为batch_size的二维矩阵
    _01_matrix = [list() for i in range(batch_size)]
    ind = 0
    for function_name, samples_line in zip(function_name_list[batch_size * (index - 1): batch_size * index],
                                           samples_line_list[batch_size * (index - 1): batch_size * index]):
        vulner_line = line_dic[function_name]
        if labels[ind] == 0:  # 样本为好样本(1: bad, 0: good)
            _01_matrix[ind] = [1 for i in range(samples_size)]  # 样本为好样本, 乘以1, 即输出所有节点的值
        else:
            for line in samples_line:
                if line == vulner_line:  # 存在漏洞行, 则矩阵相应地方设置为1
                    _01_matrix[ind].append(1)
                else:
                    _01_matrix[ind].append(0)  # 不存在漏洞行, 则矩阵相应地方设置为0
            if len(_01_matrix[ind]) < samples_size:
                for i in range(samples_size - len(_01_matrix[ind])):  # 填充大小, 保证每个矩阵大小一致
                    _01_matrix[ind].append(0)
        ind += 1
    return _01_matrix


# 得到漏洞行预测结果
def get_prediction_results(loc, function_name__test_list, line_dic, test_line_list, batch_size, index, output, labels):
    loc_list = torch.squeeze(loc).cpu().detach().numpy()
    labels_list = labels.cpu().detach().numpy()
    pred = output.squeeze()
    prediction_lines = []  # 存放每个漏洞样本的预测行
    real_vulnerable_lines = []  # 存放每个漏洞样本的真实漏洞行
    pred_correct_num = 0  # 每一个batch_size成功预测漏洞行的样本个数
    distances = 0  # 每一个batch_size预测行于实际漏洞行之间的距离(取绝对值)
    filename_list = []
    i = 0
    for function_name, test_line, pred_ind in zip(
            function_name__test_list[batch_size * (index - 1): batch_size * index],
            test_line_list[batch_size * (index - 1): batch_size * index], loc_list):
        vulner_line = line_dic[function_name]
        pred_line = 0
        if pred[i] > 0.5 and labels_list[i] == 1:
        # if pred[i] > 0.5:
            if pred_ind < len(test_line):  # 预测索引小于样本长度
                pred_line = test_line[pred_ind]
                distances += abs(int(pred_line) - int(vulner_line))
                prediction_lines.append(pred_line)  # 添加预测行到列表
                filename_list.append(function_name)
                real_vulnerable_lines.append(vulner_line)  # 添加漏洞行到列表
            if pred_line == vulner_line:
                pred_correct_num += 1
        i += 1
    return pred_correct_num, distances, prediction_lines, real_vulnerable_lines, filename_list

