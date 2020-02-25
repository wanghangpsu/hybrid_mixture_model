from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import shutil
import sys, getopt

tf.logging.set_verbosity(tf.logging.INFO)

num = 1000
step = 60000

#the structure of the CNN

def cnn_model_fn(features, labels, mode):

    #Input Layer

    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3], name="input")
    input_layer = tf.layers.batch_normalization(input_layer)

    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu,
        name="con1"
    )
    print("conv1", conv1.shape)
    conv1 = tf.layers.batch_normalization(conv1)

    #Pooling Layer #1

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2, name="conv1")
    print("pool1", pool1.shape)
    pool1 = tf.layers.batch_normalization(pool1)
    pool1 = tf.layers.dropout(
        inputs=pool1,
        rate=0.3,
        training=mode == tf.estimator.ModeKeys.TRAIN,

    )




    #Convolitional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv2"
    )
    conv2 = tf.layers.batch_normalization(conv2)

    print("conv2", conv2.shape)

    #Pooling Layer #2

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2, name="pool2")
    print("pool2", pool2.shape)
    pool2 = tf.layers.batch_normalization(pool2)
    pool2 = tf.layers.dropout(
        inputs=pool2,
        rate=0.3,
        training=mode == tf.estimator.ModeKeys.TRAIN,

    )

    #Convolutional Layer #3

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv3"
    )
    conv3 = tf.layers.batch_normalization(conv3)
    print("conv3", conv3.shape)

    #Pooling Layer #3

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name="pool3")
    print("pool3", pool3.shape)
    pool3 = tf.layers.batch_normalization(pool3)
    pool3 = tf.layers.dropout(
        inputs=pool3,
        rate=0.3,
        training=mode == tf.estimator.ModeKeys.TRAIN,

    )

    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 256], name="flat")



    dense1 = tf.layers.dense(inputs=pool3_flat, units=256, activation=tf.nn.relu, name="dense1")
    dense1 = tf.layers.batch_normalization(dense1)
    dense1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.3,
        training=mode == tf.estimator.ModeKeys.TRAIN,

    )
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu, name="dense2")
    dense2 = tf.layers.batch_normalization(dense2)

    dropout = tf.layers.dropout(inputs=dense2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")

    logits = tf.layers.dense(inputs=dropout, units=100, name="logits")

    print("logits", logits.shape)

    predictions = {
        "class": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "dense2": dense2
    }

    #predict option
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    equality = tf.equal(predictions["class"], labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32), name="training_accuracy")

   # labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=67) # depth = number of classes

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["class"]
        )
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):

    # load the data
#    train_data = np.concatenate((np.load("train_data.npy"), np.load("eval_data.npy")), axis=0)

    train_data = np.load('train_data.npy')[0:num]
    eval_data = np.load("test_data.npy")
    train_data = np.asarray(train_data, dtype=np.float32)
    eval_data = np.asarray(eval_data, dtype=np.float32)
#    train_labels = np.concatenate((np.load("train_label.npy"), np.load("eval_label.npy")), axis=0)
    train_labels = np.load('train_label.npy')[0:num]
    train_labels = np.asarray(train_labels, dtype=np.int64)
    eval_labels = np.load("test_label.npy")
    eval_labels = np.asarray(eval_labels, dtype=np.int64)

    #create the estimator

    sense_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="tmp/cnn_model"
    )

    #set up logging for predictions
    tensor_to_log = {"probabilities": "softmax_tensor", "training_acc": "training_accuracy"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensor_to_log, every_n_iter=50
    )

    #train the model

    train_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    sense_classifier.train(
        input_fn=train_input_fn,
        steps=step,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = sense_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)
    f = open('acc.txt', 'a')
    f.write(str(num))
    f.write(' ')
    f.write(str(eval_results))
    f.close()
    #shutil.rmtree('tmp')

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


if __name__=="__main__":
    #number = [1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000]
    para(sys.argv[1:])

    num = int(num)
    step = int(step)

    tf.app.run()

