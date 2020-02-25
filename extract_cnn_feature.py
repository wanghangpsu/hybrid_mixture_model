import numpy as np
import tensorflow as tf
from tf_nn_model import cnn_model_fn
import sys, getopt

classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/cnn_model")
def cnn_prob(classifier, input_data):
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": input_data}, shuffle=False)
    pred = classifier.predict(input_fn=pred_input_fn)
    pred = list(pred)
    prob_cnn = []
    for i in range(len(pred)):
        prob_cnn.append(pred[i]['dense2'])

    return np.array(prob_cnn)




def para(argv):
    global num, step
    try:
        opts, args = getopt.getopt(argv, "n:", ["number=", "step="])
    except getopt.GetoptError as err:
        print(str(err))

        sys.exit(1)

    for opt, arg in opts:

        if opt == '--number':
            num = arg
        elif opt == '--step':
            step = arg


num = 1000
step = 60000

para(sys.argv[1:])
num = int(num)

train_data = np.load('train_data.npy')[0: num]
train_data = np.asarray(train_data, dtype='float32')
test_data = np.load('test_data.npy')
test_data = np.asarray(test_data, dtype='float32')

cnn_feature_train = cnn_prob(classifier, train_data)
cnn_feature_test = cnn_prob(classifier, test_data)

np.save('cnn_feature_train.npy', cnn_feature_train)
np.save('cnn_feature_test.npy', cnn_feature_test)