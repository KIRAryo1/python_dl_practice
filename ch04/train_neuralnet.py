import numpy as np
import sys, os
sys.path.append(os.pardir)
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# Hyper Parameter
iters_num = 2
train_size = x_train.shape[0]
batch_size = 10
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # rapid version

    # update of parameter
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # log learning process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(i)

x = np.arange(0, iters_num, 1)
plt.plot(x, train_loss_list)
plt.show()
