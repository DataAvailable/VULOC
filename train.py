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
from units import data_preprocess, covert_data_to_tensor, Generate_01_matrix

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
    lr = learning_rate

    # 创建加载器
    train_loader = DataLoader(training_data, shuffle=False, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 设定存储
    clip = 5
    valid_loss_min = np.Inf
    model.train()

    train_steps = []
    train_ccr = []
    train_cost = []  # Same as training loss, but over more samples
    devel_steps = []
    devel_ccr = []

    num_correct_since_last_check = 0
    train_progress_conf = 10
    validation_progress_conf = 10
    step = 0

    no_improvement_counter = 0
    we_are_free = 0
    max_allowed_stagnations = epochs
    good_enough = 0.1  # Parameter for deciding when a model reaches sufficient accuracy
    epoch_counter = 0
    max_allowed_stagnations = 600  # Parameter for adjusting the allowed number of stagnated epochs

    while epoch_counter < epochs:
        epoch_counter += 1

        if no_improvement_counter > max_allowed_stagnations:
            print('\n\n MODEL STUCK...TERMINATING \n\n')
            break
        if we_are_free:
            print('\n\n TERMINATING... \n\n')
            break

        h = model.init_hidden(batch_size)
        # print(h)
        index = 1
        for inputs, labels in train_loader:
            step += 1
            num_correct_train = 0
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)

            # 计算每一个 batch_size 的01矩阵
            samples_size = inputs.shape[1]
            _01_matrix = Generate_01_matrix(function_name_list, samples_line_list, line_dic, labels, samples_size,
                                            batch_size, index)
            _01_matrix = torch.tensor(_01_matrix)  # list -> tensor
            _01_matrix = _01_matrix.to(device)

            model.zero_grad()
            output, h = model(inputs, h, _01_matrix, 'train')
            pred = torch.round(output.squeeze())  # rounds the output to 0/1
            correct_tensor = pred.eq(labels.float().view_as(pred))
            num_correct = np.sum(np.squeeze(correct_tensor.cpu().numpy()))
            num_correct_since_last_check += num_correct
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            index += 1

            if step % train_progress_conf == 0:
                ccr = num_correct / batch_size
                running_ccr = (num_correct_since_last_check / train_progress_conf / batch_size)
                num_correct_since_last_check = 0
                train_steps.append(step)
                train_ccr.append(running_ccr)
                train_cost.append(loss.item())

            if step % validation_progress_conf == 0:
                valid_counter = 0
                num_validated = 0
                num_correct_validation = 0
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h, _01_matrix, 'validation')
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                    valid_correct_tensor = pred.eq(labels.float().view_as(pred))
                    valid_num_correct = np.sum(np.squeeze(correct_tensor.cpu().numpy()))
                    num_correct_validation += valid_num_correct
                    valid_counter += batch_size

                devel_steps.append(step)
                devel_ccr.append(num_correct_validation / valid_counter)

                model.train()
                print("Epoch: {}".format(epoch_counter), " train_acc: {:.6f}".format(running_ccr),
                      " train_loss: {:.6f}".format(loss.item()),
                      " val_acc: {:.6f}".format(num_correct_validation / valid_counter),
                      " val_loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    no_improvement_counter = 0
                    torch.save(model.state_dict(), 'state_dict_32.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                    np.mean(
                                                                                                        val_losses)))
                    valid_loss_min = np.mean(val_losses)
                    if valid_loss_min <= good_enough:
                        print('Sufficient accuracy')
                        we_are_free = 1

                else:
                    no_improvement_counter += 1
                    print('No improvement yet, increasing counter:{}'.format(no_improvement_counter))

    # 加载最优模型
    model.load_state_dict(torch.load('./state_dict.pt'))
    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)
    print('\n\n Testing \n\n')
    print('Params: batch_size = {} epochs = {} learning_rate = {} dropout = {}'.format(batch_size, epochs,
                                                                                       learning_rate, dropout))
    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h, _01_matrix, 'test')
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))
    train_progress = {'steps': train_steps, 'ccr': train_ccr, 'cost': train_cost}
    devel_progress = {'steps': devel_steps, 'ccr': devel_ccr, 'cost': val_losses}

    # plot_progress(train_progress, devel_progress, str(name))
    # del model


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
training_data, val_data, test_data = covert_data_to_tensor(training_data, test_data, train_labels, test_labels)

print('Number of training samples: ', len(training_data))
print('Number of validation samples: ', len(val_data))
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
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, _01_matrix, dataset_type):
        batch_size = x.size(0)
        # Transform to unlimited precision
        x = x.long()

        embeds = self.embedding(x)
        embeds = torch.reshape(embeds, (batch_size, longest_sample, embedding_dim * embedding_dim))
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)     # len：2496
        out = out.view(batch_size, -1)  # 32 * 78
        if dataset_type == 'train':
            # 定位层
            out = out * _01_matrix          # 输出结果乘以01矩阵
        out = out.unsqueeze(1)          # 升维 out = [[ [ ] ]] 3维
        # k最大池化层
        kmax_result, index = kmax_pooling(out, 2, 1)

        # 全局平均池化层
        aver_pool = nn.AdaptiveAvgPool1d(1)
        out = aver_pool(kmax_result)
        out = torch.squeeze(out)        # 降维
        # out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().to(device))
        torch.nn.init.xavier_uniform_(hidden[0])
        torch.nn.init.xavier_uniform_(hidden[1])
        return hidden


# 模型参数
vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = max_seq
hidden_dim = 512
n_layers = 2
dropout = 0.2

batchsize = 32
epoch = 100
learning_rate = 0.001
# weight_decay = 0.0001

print('Vocabulary size: ', vocab_size)
print('Params: batch_size = {} epochs = {} learning_rate = {} dropout = {}'.format(batchsize, epoch, learning_rate,
                                                                                   dropout))
deep_learning(batchsize, epoch, learning_rate, 'cool_name')
