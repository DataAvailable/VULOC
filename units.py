import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc
import torch
from sklearn.manifold import TSNE


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
def data_preprocess(training_data, test_data):
    # Find index of largest sample and the longest data sample of both train and test dataset:
    longest_sample = len(training_data[0])
    longest_sample_index = 0

    for i in range(len(training_data)):
        tmp = len(training_data[i])
        if tmp > longest_sample:
            longest_sample = tmp
            longest_sample_index = i

    # Find longest sample in test data :
    # 在测试数据中找到最长的样本：
    longest_sample_test = len(test_data[0])
    longest_sample_index_test = 0

    for i in range(len(test_data)):
        tmp = len(test_data[i])
        if tmp > longest_sample_index_test:
            longest_sample_index_test = tmp
            longest_sample_index_test = i

    # Flag used as boolean
    # 用布尔值作标志
    train_longest = 1
    if longest_sample_test > longest_sample:
        longest_sample = longest_sample_test
        longest_sample_index = longest_sample_index_test
        train_longest = 0

    # 选择在最大样本中取最长的行，这并不能保证最长的行，但我们添加一些并填充其余部分
    if train_longest:
        max_seq = find_max_list(training_data[longest_sample_index])
    else:
        max_seq = find_max_list(test_data[longest_sample_index])

    for i in range(len(training_data)):
        training_data[i] = pad_sample_len(training_data[i], longest_sample)
        training_data[i] = pad_seq_len(training_data[i], max_seq)

    for i in range(len(test_data)):
        test_data[i] = pad_sample_len(test_data[i], longest_sample)
        test_data[i] = pad_seq_len(test_data[i], max_seq)

    return training_data, test_data, max_seq, longest_sample


# 加载数据
def covert_data_to_tensor(training_data, test_data, train_labels, test_labels):
    # 将数据和标签转换为 numpy 数组
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    training_data = np.asarray(training_data)
    test_data = np.asarray(test_data)

    # 我们在训练期间需要一个数据集进行验证，选择溢出一半，稍后可以调整
    # split_frac = 0.5
    # split_id = int(split_frac * len(test_data))
    # validation_data, test_data = test_data[:split_id], test_data[split_id:]
    # val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

    # 为训练、验证和测试创建 tensorDatasets
    training_data = TensorDataset(torch.from_numpy(training_data), torch.from_numpy(train_labels))
    # val_data = TensorDataset(torch.from_numpy(validation_data), torch.from_numpy(val_labels))
    test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    return training_data, test_data


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
    i = 0
    for function_name, test_line, pred_ind in zip(
            function_name__test_list[batch_size * (index - 1): batch_size * index],
            test_line_list[batch_size * (index - 1): batch_size * index], loc_list):
        vulner_line = line_dic[function_name]
        pred_line = 0
        if pred[i] > 0.5 and labels_list[i] == 1:
            if pred_ind < len(test_line):  # 预测索引小于样本长度
                pred_line = test_line[pred_ind]
                distances += abs(int(pred_line) - int(vulner_line))
                prediction_lines.append(pred_line)  # 添加预测行到列表
                real_vulnerable_lines.append(vulner_line)  # 添加漏洞行到列表
            if pred_line == vulner_line:
                pred_correct_num += 1
        i += 1
    return pred_correct_num, distances, prediction_lines, real_vulnerable_lines


# 绘制漏洞定位结果曲线图
def generate_location_results(prediction_lines_total, real_vulnerable_lines_total):
    # samples_num = [i for i in range(len(prediction_lines_total))]
    x = []
    y1 = []
    y2 = []
    for i in range(len(prediction_lines_total)):
        if i % 60 == 0:
            x.append(i)
            y1.append(int(prediction_lines_total[i]))
            y2.append(int(real_vulnerable_lines_total[i]))

    # 绘制折线图，添加数据点，设置点的大小
    # 此处也可以不设置线条颜色，matplotlib会自动为线条添加不同的颜色
    plt.plot(x, y1, alpha=1, color="c", marker='*', markersize=7)
    plt.plot(x, y2, alpha=1, color='y', marker='o', markersize=7)

    plt.title('Location Results')  # 折线图标题
    plt.xlabel('Number of samples')  # x轴标题
    plt.ylabel('Line number')  # y轴标题
    plt.grid(ls='-.')  # 绘制背景线
    plt.legend(['prediction lines', 'vulnerability lines'])
    plt.tight_layout()
    plt.savefig('Location Results')
    plt.show()

    # fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    # plt.ylim(0, 30)
    # ax1.plot(samples_num, prediction_lines_total, 'r', label='prediction line')
    # ax1.plot(samples_num, real_vulnerable_lines_total, 'g', label='vulnerable line')
    # ax1.set_xlabel('the number of samples')
    # ax1.set_ylabel('vulnerable line')
    # ax1.legend(loc='lower left', bbox_to_anchor=(0.6, 0.52), framealpha=1.0)
    #
    # plt.title('Location results')
    # fig.tight_layout()
    # plt.savefig('location_results')
    # plt.show()


# ROC图
def draw_roc(pred_list, labels_list):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(labels_list, pred_list)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_list, pred_list)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6), dpi=150)
    lw = 2
    plt.plot(
        fpr[1],
        tpr[1],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc[1],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC Curve')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig('ROC')
    plt.show()


# t-SNE散点图
def draw_tsne(data, label):
    data = np.array(data)
    label = label  # shape(360,)
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features   # shape(360, 64) 2, 360, 64


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if i % 5 == 0:
            if label[i] == 1:    # 存在漏洞
                plt.text(data[i, 0], data[i, 1], 'x',
                         color='hotpink',
                         fontdict={'size': 11})
            else:
                plt.text(data[i, 0], data[i, 1], 'o',
                         color='lightseagreen',
                         fontdict={'size': 11})
    plt.xticks([])
    plt.yticks([])
    plt.savefig('t-SNE')
    plt.title(title)
    return fig
