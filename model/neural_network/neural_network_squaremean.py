import matplotlib.pyplot as plt
import math
import random
import numpy as np
import copy

#sigmoid function
def g(value):
	return 1.0/(1+math.exp(-value))

#feed forward propagation
def ffp (x_vec, bias, weights, nodes):
	print("called")
	temp_vec = x_vec
	for i in range(len(nodes)):
		for j in range(len(nodes[i])):
	#		print(" i %i j%i g(bias[i][j] + np.dot(temp_vec,weights[i][j]) %f" %(i,j, g(bias[i][j] + np.dot(temp_vec,weights[i][j]))))
			nodes[i][j] = g(bias[i][j] + np.dot(temp_vec,weights[i][j]))
		temp_vec = nodes[i]	
	return temp_vec
	
#function used to monitor the error progression
def cost_general(samples, y_vec, bias, weights, nodes):  
	error_acum = 0
	for i in range(len(samples)):
		hyp = ffp(samples[i], bias, weights, nodes)
		for k in range(len(y_vec[i])):
			print(" samples x1 %i  x2 %i  hyp %f y_vec[%i][%i] %f" %(samples[i][0],samples[i][1], hyp[k],i,k,y_vec[i][k]))
			error = .5*(y_vec[i][k]-hyp[k])**2
			error_acum = error_acum + error
	return error_acum

#cost function for an intance
def cost_single(x_vec, y_vec, bias, weights, nodes):
	error = [0]*y_vec
	hyp = ffp(x_vec, bias, weights, nodes)
	for k in range(len(y_vec)):
		print("k %i" %(k))
		error[k] = (y_vec[k]-hyp[k])*-1.0
	return error

	
#intermediate step to get the matrixes
def generate_deltas(weights,  deltas, nodes, error):
	layers = len(deltas)-1
	for k in range(len(deltas[layers])-1,-1,-1):
#		print(" %f * %f " % (error[k],nodes[layers][k]*(1 - nodes[layers][k])))
		deltas[layers][k] = error[k] *nodes[layers][k]*(1- nodes[layers][k])
	for l in range(layers-1,-1,-1):
		for j in range(len(deltas[l])-1,-1,-1):
			a = np.array(weights[l+1])
			a = a.T
			deltas[l][j] = np.dot(deltas[l+1],a[j])*nodes[l][j]*(1-nodes[l][j])
	return deltas

def generate_gradients(sample, gradients, nodes, deltas):
	ng = 1;
	for l in range(len(gradients)):
		for j in range(len(gradients[l])):
			for k in range(len(gradients[l][j])):
				if(l == 0): #use sample
					gradients[l][j][k] =  sample[k]*deltas[l][j]
					#print("%i gradient[%i] [%i] [%i] = samples[%i] %f * delta[%i][%i] %f = %f" %(ng,l,j,k,k, sample[k],l,j, deltas[l][j], gradients[l][j][k]))
				else:
					gradients[l][j][k] =  nodes[l-1][k]*deltas[l][j]
					#print("%i gradient[%i] [%i] [%i] = nodes[%i][%i] %f * delta[%i][%i] %f = %f" %(ng,l,j,k, l-1,k, nodes[l-1][k], l,j, deltas[l][j], gradients[l][j][k]))
				ng = ng +1
	return gradients

def back_propagation(sample, y_vec, bias, weights, nodes):
	deltas = copy.deepcopy(nodes)
	gradients = copy.deepcopy(weights)
	

	clean(gradients) 
	clean(deltas)

	error = cost_single (sample, y_vec, bias, weights, nodes)
	deltas = generate_deltas (weights,deltas,nodes, error)
	gradients = generate_gradients(sample, gradients, nodes, deltas)


	for l in range(len(gradients)):
		for j in range(len(gradients[l])):
			for k in range(len(gradients[l][j])):
#				print("w %i  = %f - .5 * %f = %f" %(i,weights[l][j][k],gradients[l][j][k],weights[l][j][k] - gradients[l][j][k]*.5))
				weights[l][j][k] = weights[l][j][k] - gradients[l][j][k]*.5
	return gradients

#sets a mat to 0s
def clean(mat):
	mat = mat*0

def random_weights(samples, nodes):
	layers = len(nodes)
	weights = [0]*layers
	for i in range(layers):
		weights[i] = [0] * len(nodes[i])
		for j in range(len(weights[i])):
			if(i ==0):
				weights[i][j] = [0] * len(samples[0])
			else:
				weights[i][j] = [0] * len(nodes[i-1])
			for k in range(len(weights[i][j])):
				weights[i][j][k] = random.random()
	return weights
	
def random_bias(nodes):
	layers = len(nodes)
	bias = [0]*layers
	for i in range(layers):
		bias[i] = [0] * len(nodes[i])
		for j in range(len(bias[i])):
			bias[i][j] = random.random()
	return bias

#format of inputs

#samples[sample][X]->([sample1[x0,x1,x2..Xn],sample2[x0,x1,x2..Xn]])
#single node class[sample][class]->([sample1[class1], sample2[class2]])
#several nodes class[sample][class]->([sample1[class1, class2, classn], sample2[class1, class2, classn]])
#weight[layer][node][weight] -> ([layer1[node1[weight1, weight2], node2[weight1]], layer2[node[weight]]])
#nodes[layer][node]  -> ([layer1[node1,node2], layer2[node1]]
#deltas[layer][node]  -> ([layer1[node1,node2], layer2[node1]]
#bias[layer][node]  -> ([layer1[node1,node2], layer2[node1]]

"""
# Random Func. Example 1
samples = np.array([[.05,.1]]) 
y_vec = np.array([[.01,.99]])
weights = np.array([[[0.15, 0.2],[ 0.25, 0.3]],[[0.4, 0.45], [0.5, 0.55]]])
nodes = np.array([[0.0, 0.0],[0.0, 0.0]])
deltas = np.array([[0.0, 0.0],[0.0, 0.0]])
bias = np.array([[.35,.35],[.6,.6]])
"""
"""
# OR
samples = np.array([[1.,1.],[0.,1.],[1.,0.],[0.,0.]])  
y_vec = np.array([[1.],[1.],[1.],[0.]]) 
nodes = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
deltas = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
weights = np.array(random_weights(samples,nodes))
bias = np.array(random_bias(nodes))
"""
"""
# AND
samples = np.array([[1.,1.],[0.,1.],[1.,0.],[0.,0.]])  
y_vec = np.array([[1.],[0.],[0.],[0.]]) 
nodes = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
deltas = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
weights = np.array(random_weights(samples,nodes))
bias = np.array(random_bias(nodes))
"""

"""
# XNOR
samples = np.array([[1.,1.],[0.,1.],[1.,0.],[0.,0.]])  
y_vec = np.array([[1.],[0.],[0.],[1.]]) 
nodes = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
deltas = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
weights = np.array(random_weights(samples,nodes))
bias = np.array(random_bias(nodes))
"""

# XOR
samples = np.array([[1.,1.],[0.,1.],[1.,0.],[0.,0.]]) 
y_vec = np.array([[0.],[1.],[1.],[0.]]) 
nodes = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
deltas = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
weights = np.array(random_weights(samples,nodes))
bias = np.array(random_bias(nodes))


#print("gradcheck")
#print(gradCheck(samples, y_vec, bias, weights, nodes))
errors = np.zeros(10000)

for i in range(10000):
	index =i%len(samples) #assign training examples from sample set
#	print(samples[index])
#	print(y_vec[index])
	back_propagation(samples[index], y_vec[index], bias, weights, nodes)
	error = cost_general(samples, y_vec, bias, weights, nodes)
	errors[i] = error
	print("%i error %f" %(i, error))
	if(error < .01):
		print("happy day (crying)")
		break
print(weights)
x_axis = range (1, len(errors)+1)
plt.plot(x_axis, errors, 'b', label='training error')
plt.title('Training Error Graph')
plt.legend()
plt.show()