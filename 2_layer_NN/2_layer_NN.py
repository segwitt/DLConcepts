import keras
import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train,10)
print(y_train[1])


x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32')
x_train /= 255
x_test /= 255

#defining the forward pass
def softmax(y):
  return np.exp(y) / np.sum(sum(np.exp(y)))


def forward(W, x, b):
  # x is input and y is output
  o = np.dot(W.T, x) + b
  return softmax(o)

def gradient(W, x, b, yh):
  yp = forward(W, x, b)
  yp -= yh
  grad = np.dot(x.reshape(x.shape[0],1), yp.reshape(1, yp.shape[0]))
  return grad, yp

def update(W, b, grad, grad_bias, learning_rate = 0.0001):
  W -= learning_rate * grad
  b -= grad_bias
  return None

def backward(W, b, x, y):
  grad, grad_bias = gradient(W, x, b, y)
  update(W, b, grad, grad_bias)
  return None

def test(W, b, x_test, y_test):
  cnt = 0
  for (x, y) in zip(x_test, y_test):
    yp = forward(W, x, b)
    if yp.argmax() == y.argmax():
      cnt += 1
  return cnt / y_test.shape[0]

def train(W, b, x_train, y_train, epochs=20):
  for _ in range(epochs):
    for (x, y) in zip(x_train, y_train):
      backward(W, b, x, y)
    acc = test(W, b, x_test, y_test)
    print('val acc', acc)
      
  return None

def main():
	W = np.random.randn(784,10)
	b = np.random.randn(10,).astype('float32')

	train(W, b, x_train, y_train)
	return None



if __name__ == '__main__':
	main()
