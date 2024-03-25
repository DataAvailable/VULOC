import matplotlib.pyplot as plt
from units import data_preprocess
from loading_data import load_dataset
from collections import Counter
import numpy as np

words, training_data, test_data, train_labels, test_labels, function_name_list, function_name_test_list, samples_line_list, test_line_list, line_dic = load_dataset()

words = sorted(words, key=words.get, reverse=True)
words = ['_PAD', '_UNK'] + words
word2idx = {o: i for i, o in enumerate(words)}

for m in range(len(test_data)):
    for n, sentence in enumerate(test_data[m]):
        test_data[m][n] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i in range(len(training_data)):
    for j, sentence in enumerate(training_data[i]):
        training_data[i][j] = [word2idx[word] if word in word2idx else 1 for word in sentence]
print(training_data)


len_list = []
for i in range(len(training_data)):
    len_list.append(len(training_data[i]))
print(len_list)
list2 = list(set(len_list))
print(list2)
d2 = Counter(len_list)
print(d2)
list3 = []
for i in range(len(list2)):
    list3.append(d2[list2[i]])

# plt.subplot(1, 1, 1)
fig = plt.figure()
plt.figure(figsize=(8, 6), dpi=200)
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
x = np.array(list2)
print('x: ', x)
y = np.array(list3)
plt.bar(x, y, width=0.5, align="center")
# plt.title("The histogram of vectors length distribution", loc="center", fontsize=18)
num = 0
little = 0
big = 0
for a, b in zip(x, y):
    # if a % 5 == 0:
    #     plt.text(a, b, b, ha='center', va='center', fontsize=12)
    num = num + b
    if a <= 100:
        little = little + b
    else:
        big = big + b

print('总数量: ', num)
print('长度小于100: ', little)
print('长度大于100: ', big)

plt.ylabel("The number of vectors corresponding to CoRS", fontsize=12, family='Times New Roman')
plt.xlabel("The length of vectors corresponding to CoRS", fontsize=12, family='Times New Roman')

plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.savefig("distribution_graph.jpg")
plt.show()
