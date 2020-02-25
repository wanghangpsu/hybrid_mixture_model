import numpy as np



def predict_t(X, W, b):
    scores = np.exp(np.dot(X, W) + b)
    probs = scores / np.sum(scores, axis=1, keepdims=True)
    return probs


def cost_t(X, W, b, y):
    cost = 0
    p = predict_t(X, W, b)
    n = np.shape(X)[0]
    for i in range(n):
        cost -= np.log(p[i][int(y[i])])
    return cost / n

def compute_gradient_t(X, W, b, y):
    p = predict_t(X,W,b)
    one = np.zeros_like(p)
    n = np.shape(X)[0]
    for i in range(np.shape(y)[0]):
        one[i][int(y[i])] = 1
    df = p - one
    d_W = np.dot(X.T, df) / n
    c = np.ones([1, np.shape(y)[0]])
    d_b = np.dot(c, df) / n
    return (d_W, d_b)

if __name__ == '__main__':

    REG = 0
    NUM_GROUP = 100
    STEPS = 1500
    STEP_SIZE = 0.05
    number = [500,1000,5000,10000,20000,30000,40000,50000]


    #loading the data
#    x_all = np.concatenate((np.load('train_features.npy')[0:10000], np.load("test_features.npy")[0:5000]), axis=0)     # the input data is
#    label = np.concatenate((np.load('train_label.npy')[0:10000], np.load('eval_label.npy')), axis=0)
    f = open('acc1.txt', 'w')
    for num in number:

        x_all = np.load('train_features.npy')[0: num]

        label = np.load('train_label.npy')[0: num]

        FEATURE_LENGTH = x_all.shape[1]
        W = np.random.random([FEATURE_LENGTH, NUM_GROUP])-0.5
        b = np.random.random([1, NUM_GROUP])-0.5


        for i in range(STEPS):
            dW, db = compute_gradient_t(x_all, W, b, label)
            W -= STEP_SIZE * dW
            b -= STEP_SIZE * db
            loss = cost_t(x_all, W, b, label)
            print("iteration %d: loss %f" % (i, loss))


        x_test = np.load('test_features.npy')

        test_label = np.load('test_label.npy')



        prob = predict_t(x_test, W, b)

        predicted_class = np.argmax(prob, axis=1)

        acc = np.mean(predicted_class == test_label)

        print(acc)
        f.write(str(num))
        f.write(' ')
        f.write(str(acc))
        f.write('\n')

    f.close()



