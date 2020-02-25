import numpy as np


NUM = 10

train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
train_feature = np.load('train_features.npy')

count = [0] * 10
index = []

i = 0
j = 0
for label in train_label:
    print(label)
    if count[int(label)] < NUM:
        index.append(j)
        count[int(label)] += 1

        i = i + 1
    j += 1
    if i == 10 * NUM:
        break


small_data = train_data[index]
small_label = train_label[index]
small_feature = train_feature[index]




