import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import re
import nltk
import os
# nltk.download('punkt')
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loading_data import load_dataset
from sklearn.manifold import TSNE
from units import data_preprocess, covert_data_to_tensor, get_prediction_results, generate_location_results, \
    draw_roc, draw_tsne, plot_embedding

plt.style.use('seaborn')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
# Hardcoded for my personal archatecture
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    print('gpu')
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
else:
    print('cpu')
    device = torch.device("cpu")


# Function for doing deep learning
def deep_learning(batch_size, epochs, learning_rate, name):

    model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout)
    model.to(device)
    print(model)
    batch_size = batch_size
    epochs = epochs
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
    criterion = nn.BCELoss()
    model.train()

    # 加载最优模型
    model.load_state_dict(torch.load('state_dict.pt'))
    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)
    print('\n Testing... \n')
    print('Params: batch_size = {} epochs = {} learning_rate = {} dropout = {}'.format(batch_size, epochs,
                                                                                       learning_rate, dropout))
    model.eval()
    pred_list = []
    labels_list = []
    pred_labels = []
    index = 1       # 索引, 表示第几个batch_size
    pred_lines_correct = 0
    total_distances = 0
    prediction_lines_total = []
    real_vulnerable_lines_total = []
    '''
    '''
    preds_tsne = []
    labels_tsne = []
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h, loc, out_tsne = model(inputs, h)
        pred_list.extend(output.cpu().detach().numpy())     #
        # 获取预测行
        pred_correct_num, distances, prediction_lines, real_vulnerable_lines = \
            get_prediction_results(loc, function_name_test_list, line_dic, test_line_list, batch_size, index, output, labels)
        # 计算定位正确的样本数
        pred_lines_correct += pred_correct_num
        total_distances += distances
        # 保存预测结果和真实结果
        prediction_lines_total.extend(prediction_lines)
        real_vulnerable_lines_total.extend(real_vulnerable_lines)
        ### 保存绘制tsne图所需参数
        preds_tsne.extend(out_tsne.cpu().detach().numpy())
        labels_tsne.extend(labels.cpu().detach().numpy())
        ###
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

        pred_labels.extend(pred.cpu().detach().numpy())
        labels_ndarray = labels.cpu().detach().numpy()
        labels_list.extend(labels_ndarray)
        index += 1

    # 计算FPR, FNR, Pr, Re, F1
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for pre, lab in zip(pred_labels, labels_list):
        if pre == 1.0 and lab == 1:
            tp += 1
        elif pre == 1.0 and lab == 0:
            fp += 1
        elif pre == 0.0 and lab == 0:
            tn += 1
        else:
            fn += 1
    fprs = fp / (fp + tn)
    fnrs = fn / (tp + fn)
    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    f1 = (2 * pr * re) / (pr + re)
    acc = num_correct / len(test_loader.dataset)

    # 绘制定位差值结果图
    generate_location_results(prediction_lines_total, real_vulnerable_lines_total)
    # 绘制ROC图
    draw_roc(pred_list, labels_list)

    # 绘制t-SNE散点图
    data, label, n_samples, n_features = draw_tsne(preds_tsne, labels_tsne)
    print('Computing t-SNE embedding...')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label, 't-SNE')
    plt.show()

    print("Test loss: {:.3f} Test accuracy: {:.3f}%".format(np.mean(test_losses), acc * 100))
    # print("Test loss: " '0.119 ' "Test accuracy: " '98.930%')
    print('FPR = {:.3f} FNR = {:.3f} Pr = {:.3f} Re = {:.3f} F1 = {:.3f}'.format(fprs, fnrs, pr, re, f1))
    print("Correct location numbers: {} total numbers: {} location rate: {}".
          format(pred_lines_correct, tp, pred_lines_correct/tp))
    print("average distance: ", total_distances/tp)


print(torch.__version__)

# 加载数据集
words, training_data, test_data, train_labels, test_labels, function_name_list, function_name_test_list, samples_line_list, test_line_list, line_dic = load_dataset()

# 根据出现次数对单词进行排序，出现最多的单词排在第一位
words = sorted(words, key=words.get, reverse=True)
words = ['_PAD', '_UNK'] + words
word2idx = {o: i for i, o in enumerate(words)}
# idx2word = {i:o for i,o in enumerate(words)}

# 查找映射字典并为各个单词分配索引
for i in range(len(training_data)):
    for j, sentence in enumerate(training_data[i]):
        # 查找映射字典并为各个单词分配索引
        training_data[i][j] = [word2idx[word] if word in word2idx else 1 for word in sentence]

for i in range(len(test_data)):
    for j, sentence in enumerate(test_data[i]):
        # 尝试使用 '_UNIK' 表示看不见的词，可以稍后更改
        test_data[i][j] = [word2idx[word] if word in word2idx else 0 for word in sentence]

# 数据预处理: 填充短样本, 填充短代码行
training_data, test_data, max_seq, longest_sample = data_preprocess(training_data, test_data)

# 加载数据, 将数据加载为tensor格式
# training_data, val_data, test_data = covert_data_to_tensor(training_data, test_data, train_labels, test_labels)
training_data, test_data = covert_data_to_tensor(training_data, test_data, train_labels, test_labels)

print('Number of training samples: ', len(training_data))
print('Number of testing samples: ', len(test_data))


def kmax_pooling(data, dim, k):        # k最大池化
    index = data.topk(k, dim=dim)[1].sort(dim=dim)[0]
    kmax_result = data.gather(dim, index)
    return kmax_result, index


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * embedding_dim, hidden_dim, n_layers, dropout=dropout,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # Transform to unlimited precision
        x = x.long()

        embeds = self.embedding(x)
        embeds = torch.reshape(embeds, (batch_size, longest_sample, embedding_dim * embedding_dim))
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)     # len：2496
        out = out.view(batch_size, -1)  # 32 * 78
        out_tsne = out  # 用于绘制tsne图
        out = out.unsqueeze(1)
        # k最大池化层
        kmax_result, loc = kmax_pooling(out, 2, 1)
        # 全局平均池化层
        aver_pool = nn.AdaptiveAvgPool1d(1)
        out = aver_pool(kmax_result)
        out = torch.squeeze(out)        # 降维
        # out = out[:, -1]
        return out, hidden, loc, out_tsne

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        torch.nn.init.xavier_uniform_(hidden[0])
        torch.nn.init.xavier_uniform_(hidden[1])
        return hidden


# 模型参数
vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = max_seq
hidden_dim = 512
n_layers = 2
dropout = 0.4

batchsize = 32
epoch = 100
learning_rate = 0.001
# weight_decay = 0.0001

print('Vocabulary size: ', vocab_size)
print('Params: batch_size = {} epochs = {} learning_rate = {} dropout = {}'.format(batchsize, epoch, learning_rate,
                                                                                   dropout))
deep_learning(batchsize, epoch, learning_rate, 'cool_name')
