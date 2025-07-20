import numpy as np # import Numpy library to generate 

weights = np.around(np.random.uniform(size=6), decimals= 2)# initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases
weights = np.around(np.random.uniform(size=6), decimals= 2)# initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases

print(weights)
print(biases)

x_1 = 0.5 # input 1
x_2 = 0.85 # input 2

print('x1 is {} and x2 is {}'.format(x_1, x_2))

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
z_12 = x_2 * weights[2] + x_1* weights[3] + biases[0]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_12))