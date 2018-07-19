import sys, os
sys.path.append(os.pardir)
# sys.path.append(os.path)
# sys.path.append(os.path.pardir)
# print(sys.path)
# print(os.path.exists("./dataset/mnist.py"))
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
