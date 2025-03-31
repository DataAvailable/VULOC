import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from loading_data import load_dataset
from units import data_preprocess, covert_data_to_tensor, get_prediction_results


def checks():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print('gpu')
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0))
    else:
        # print('cpu')
        device = torch.device("cpu")
    return device


# Function for doing deep learning
def deep_learning(poc):
    device = checks()
    test_data, max_seq, longest_sample, function_name_test_list, line_dic, test_line_list = preprocess()

    vocab_size = 5590
    output_size = 1
    embedding_dim = max_seq
    hidden_dim = 512
    n_layers = 2
    dropout = 0.2

    batch_size = 2
    epochs = 100
    learning_rate = 0.001
    # weight_decay = 0.0001

    # print('Vocabulary size: ', vocab_size)
    # print('Params: batch_size = {} epochs = {} learning_rate = {} dropout = {}'.format(batch_size, epochs, learning_rate,
    #                                                                                    dropout))
    model = SentimentNet(vocab_size, output_size, longest_sample, embedding_dim, hidden_dim, n_layers, dropout, device)
    model.to(device)
    # print(model)

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
    criterion = nn.BCELoss()
    model.train()

    # 加载最优模型
    # model.load_state_dict(torch.load('./state_dict.pt'))
    model.load_state_dict(torch.load('./model/' + poc, map_location='cpu'))
    print(poc)
    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)
    print('\n 检测中, 请稍候... \n')
    # print('Params: batch_size = {} epochs = {} learning_rate = {}'.format(batch_size, epochs,
    #                                                                       learning_rate))
    model.eval()
    pred_list = []
    labels_list = []
    pred_labels = []
    index = 1  # 索引, 表示第几个batch_size
    pred_lines_correct = 0
    total_distances = 0
    prediction_lines_total = []
    real_vulnerable_lines_total = []
    filename_list_total = []
    '''
    '''
    preds_tsne = []
    labels_tsne = []
    for inputs, labels in test_loader:
       # print(inputs)
       # print(labels)
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h, loc = model(inputs, h)
        pred_list.extend(output.cpu().detach().numpy())  #
        # 获取预测行
        pred_correct_num, distances, prediction_lines, real_vulnerable_lines, filename_list = \
            get_prediction_results(loc, function_name_test_list, line_dic, test_line_list, batch_size, index, output,
                                   labels)
        # 计算定位正确的样本数
        pred_lines_correct += pred_correct_num
        total_distances += distances

        # 保存预测结果和真实结果
        prediction_lines_total.extend(prediction_lines)
        real_vulnerable_lines_total.extend(real_vulnerable_lines)
        filename_list_total.extend(filename_list)

        ###
        preds_tsne.append(output.cpu().detach().numpy())
        labels_tsne.append(labels.cpu().detach().numpy())
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
    # tp = 0
    # fp = 0
    # tn = 0
    # fn = 0
    # for pre, lab in zip(pred_labels, labels_list):
    #     if pre == 1.0 and lab == 1:
    #         tp += 1
    #     elif pre == 1.0 and lab == 0:
    #         fp += 1
    #     elif pre == 0.0 and lab == 0:
    #         tn += 1
    #     else:
    #         fn += 1
    # fprs = fp / (fp + tn)
    # fnrs = fn / (tp + fn)
    # pr = tp / (tp + fp)
    # re = tp / (tp + fn)
    # f1 = (2 * pr * re) / (pr + re)
    # acc = num_correct / len(test_loader.dataset)

    # print("Test loss: {:.3f} Test accuracy: {:.3f}%".format(np.mean(test_losses), acc * 100))
    # print('FPR = {:.3f} FNR = {:.3f} Pr = {:.3f} Re = {:.3f} F1 = {:.3f}'.format(fprs, fnrs, pr, re, f1))
    # print("Correct location numbers: {} total numbers: {} location rate: {}".
    #       format(pred_lines_correct, tp, pred_lines_correct / tp))
    # print("average distance: ", total_distances / tp)
    #print(filename_list_total)
    #print(prediction_lines_total)
    result = []
    for filename, pred_line in zip(filename_list_total, prediction_lines_total):
        output = '{}:{}'.format(filename, pred_line)
        result.append(output)
        #print('程序 {} 的漏洞行： {}'.format(filename, pred_line))
    result = s="\n".join(result)
    return(result)
    #return result
    #     result.append('程序 ' + filename + ' 的漏洞行： ' + pred_line)
    # return result


def preprocess():
    print(torch.__version__)
    # 加载数据集
    words, test_data, test_labels, function_name_list, function_name_test_list, test_line_list, line_dic = load_dataset()
    # 根据出现次数对单词进行排序，出现最多的单词排在第一位
    words = sorted(words, key=words.get, reverse=True)
    words = ['_PAD', '_UNK'] + words
    word2idx = {o: i for i, o in enumerate(words)}
    # idx2word = {i:o for i,o in enumerate(words)}

    # 查找映射字典并为各个单词分配索引
    for i in range(len(test_data)):
        for j, sentence in enumerate(test_data[i]):
            # 尝试使用 '_UNIK' 表示看不见的词，可以稍后更改
            test_data[i][j] = [word2idx[word] if word in word2idx else 0 for word in sentence]
    # 数据预处理: 填充短样本, 填充短代码行
    test_data, max_seq, longest_sample = data_preprocess(test_data)
    # 加载数据, 将数据加载为tensor格式
    # training_data, val_data, test_data = covert_data_to_tensor(training_data, test_data, train_labels, test_labels)
    test_data = covert_data_to_tensor(test_data, test_labels)
    # print('Number of testing samples: ', len(test_data))
    return test_data, max_seq, longest_sample, function_name_test_list, line_dic, test_line_list


def kmax_pooling(data, dim, k):  # k最大池化
    index = data.topk(k, dim=dim)[1].sort(dim=dim)[0]
    kmax_result = data.gather(dim, index)
    return kmax_result, index


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, longest_sample, embedding_dim, hidden_dim, n_layers, dropout, device):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.longest_sample = longest_sample
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * embedding_dim, hidden_dim, n_layers, dropout=dropout,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # Transform to unlimited precision
        x = x.long()

        embeds = self.embedding(x)
        embeds = torch.reshape(embeds, (batch_size, self.longest_sample, self.embedding_dim * self.embedding_dim))
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)  # len：2496
        out = out.view(batch_size, -1)  # 32 * 78
        out = out.unsqueeze(1)
        # k最大池化层
        kmax_result, loc = kmax_pooling(out, 2, 1)
        # 全局平均池化层
        aver_pool = nn.AdaptiveAvgPool1d(1)
        out = aver_pool(kmax_result)
        out = torch.squeeze(out)  # 降维
        # out = out[:, -1]
        return out, hidden, loc

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        torch.nn.init.xavier_uniform_(hidden[0])
        torch.nn.init.xavier_uniform_(hidden[1])
        return hidden


# deep_learning()
