# code for processing the image data
# By Hang Wang

import numpy as np
import glob, os
import random
import sys, getopt
from PIL import Image
from natsort import natsorted

SIZE = 200
TRAINING_SAMPLE = 12000
TEST_SAMPLE = 3000
path = 'Images/'

def find_folder(path):
    data = []
    label = []
    folders = os.listdir(path)
    folders = natsorted(folders, key=lambda y: y.lower())
    for f in folders:
        folder_path = path + f
        l = folders.index(f)
        data, label = load_image(data, label, folder_path, l)
        print('folder: ' + f + ' finished...')
    print('finish!')
    return data, label


def load_image(data, label, folder_path, which_label):
    images_path = glob.glob(folder_path + '/' + '*.jpg')
    images_path = natsorted(images_path, key=lambda y: y.lower())
    for image in images_path:
        im = Image.open(image)
        im = im.resize([SIZE, SIZE])
        im_array = np.array(im)
        if im_array.shape == (SIZE, SIZE, 3):
            label.append(which_label)
            data.append(im_array)
    return data, label

def usage():
    """
    usage : python3 load_image.py

    options

    --path <the input image path>               default: Image/
    --size <the out image size>                 default: 200
    --train <the number of training samples>    default: 12000
    --test <the number of test samples>         default: 3000

    """


def para(argv):
    global SIZE, TRAINING_SAMPLE, TEST_SAMPLE, path
    try:
        opts, args = getopt.getopt(argv,"p:",["path=","size=", "train=", "test=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        print(usage.__doc__)
        sys.exit(1)

    for opt, arg in opts:
        if opt == '--help':
            print(usage.__doc__)
            sys.exit()
        elif opt in('-p', '--path'):
            path = arg
        elif opt == '--size':
            SIZE = arg
        elif opt == '--train':
            TRAINING_SAMPLE = arg
        elif opt =='--test':
            TEST_SAMPLE = arg






if __name__ == "__main__":

    para(sys.argv[1:])

    data, label = find_folder(path)

    samples = list(range(len(label)))
    train_sample = random.sample(samples, TRAINING_SAMPLE)
    #train_sample.sort()
    samples = [x for x in samples if x not in train_sample]
    test_sample = random.sample(samples, TEST_SAMPLE)
    #test_sample.sort()
    data = np.array(data, dtype=np.float16)
    label = np.array(label, dtype=np.int32)

    train_data = data[train_sample, :, :, :]
    test_data = data[test_sample, :, :, :]
    train_label = label[train_sample]
    test_label = label[test_sample]


    np.save('train_data.npy', train_data)
    np.save('train_label.npy', train_label)
    np.save('test_data.npy', test_data)
    np.save('test_label.npy', test_label)


