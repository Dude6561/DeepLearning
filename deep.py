import numpy as np # import Numpy library to generate 
weights = np.around(np.random.uniform(size=15), decimals= 2)# initialize the weights for first layer

weightsS = np.around(np.random.uniform(size=6), decimals= 2)# initialize the weights for second layer
weightsT = np.around(np.random.uniform(size=6), decimals= 2)# initialize the weights for third layer
weightsl = np.around(np.random.uniform(size=3), decimals= 2)# initialize the weights for last


print(weights)

## for the first input layer
x_1 = 0.5
x_2 = 0.85
x3 = 0.23
x_4 = 0.69
x_5 = 0.89

print('x1 is {} and x2 is {}'.format(x_1, x_2))
#for hidden layer
z_11 = x_1 * weights[0] + x_2 * weights[1] + x3 * weights[2] + x_4*weights[3] + x_5 * weights[4]
z_12 = x_1 * weights[5] + x_2 * weights[6] + x3 * weights[7] + x_4*weights[8] + x_5 * weights[9]
z_13 = x_1 * weights[10] + x_2 * weights[11] + x3 * weights[12] + x_4*weights[13] + x_5 * weights[14]



#for second hidden layer
z_21 = z_11 * weightsS[0] + z_12 * weightsS[1] + z_13 * weights [2]
z_22 = z_11 * weightsS[3] + z_12 * weightsS[4] + z_13 * weights [5]
 
 
z_31 = z_21* weightsT[0] + z_22 * weightsT[1]
z_32 = z_21* weightsT[2] + z_22 * weightsT[3]
z_33 = z_21* weightsT[4] + z_22 * weightsT[5]

z2= z_31 * weightsl[0] + z_32 * weights[1] + z_32* weights[2]

print('last output without predictation {}'.format(z2))






#### Use the *initialize_network* function to create a network that:

# 1. takes 5 inputs
# 2. has three hidden layers
# 3. has 3 nodes in the first layer, 2 nodes in the second layer, and 3 nodes in the third layer
# 4. has 1 node in the output layer

# Call the network **small_network**.
