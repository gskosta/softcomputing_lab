import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_of_sigmoid(x):
    return x * (1 - x)


inputs = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [1, 1, 1]])

actual_outputs = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

weights = 2 * np.random.random((3, 1)) - 1

print('Random weights we\'re starting out with:')
print(weights)

for x in range(100000):
    input_layer = inputs
    outputs = sigmoid(np.dot(input_layer, weights))
    error = actual_outputs - outputs

    adjustments = error * derivative_of_sigmoid(outputs)

    weights += np.dot(input_layer.T, adjustments)

print('Weights after adjustments: ')
print(weights)

for idx,val in enumerate(outputs):
    if val > 0.9:
        outputs[idx] = 1
    else:
        outputs[idx] = 0
        
print('Obtained outputs: ')
print(outputs)