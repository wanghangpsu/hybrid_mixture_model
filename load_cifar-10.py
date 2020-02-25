import numpy as np
import pickle
from PIL import Image



num_batches = 5


def load_cfar10_batch(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path + '/train', mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['fine_labels']

    return features, labels

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_test_batch(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path + '/test' , mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['fine_labels']

    return features, labels


if __name__ == "__main__":

    train_data, train_label = load_cfar10_batch("cifar-100-python")




    test_data, test_label = load_test_batch("cifar-100-python")


    for i in range(train_data.shape[0]):
        a = train_data[i]
        image = Image.fromarray(a)
        out_name = "cifar_images/train/" + '{0:05}'.format(i) + '.jpg'
        image.save(out_name)

    print('training image saved')

    for j in range(test_data.shape[0]):
        a = test_data[j]
        image = Image.fromarray(a)
        out_name = 'cifar_images/test/' + '{0:05}'.format(j) + '.jpg'
        image.save(out_name)
    print('test image saved')
    
    train_data = np.asarray(train_data, dtype=np.float16)
    test_data = np.asarray(test_data, dtype=np.float16)
    train_label = np.asarray(train_label, dtype=np.int32)
    test_label = np.asarray(test_label, dtype=np.int32)




    np.save('train_data.npy', train_data)
    np.save('train_label.npy', train_label)



    np.save('test_data.npy', test_data)
    np.save('test_label.npy', test_label)


