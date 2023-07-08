import numpy as np

# ignore bias
class layer:
	def __init__(self, weights, acvfn=None, deriv_acvfn=None):
		# bias in R^N where N is width of layer
		# weights are NxM matrix where M is width of previous layer
		# acvfn is the actiavtion function
		self.weights = weights
		self.out = None
		self.activation = None
		self.acvfn = acvfn
		self.deriv_acvfn = deriv_acvfn

	def compute(self, x):
		# apply weights
		self.activation = np.matmul(self.weights, x)
		# add bias
		# x2 = x1 + self.bias
		# apply activation function
		# print(x1)
		self.out = self.acvfn(self.activation)
		# print(self.out)
		return self.out

def init_random_layer(indim, outdim, acvfn=None):
	# pattern matching by richard duda said this was the right thing to do
	weights = (2/np.sqrt(indim))*(np.random.rand(outdim, indim) - (1/2) * np.ones((outdim,indim)))
	return layer(weights,acvfn=acvfn)

def init_random_network(architecture, acvfn, deriv_acvfn, loss, deriv_loss, linear_on_final = False, momentum_coef = None):
	# architecture is array of integers corresponsing width to sequential layers
	layers = []
	# indim = architecture[0]
	# outdim = architecture[1]
	# layers.append(init_random_layer(indim,outdim))
	for i in range(0,len(architecture)-1):
		indim = architecture[i]
		outdim = architecture[i+1]
		layers.append(init_random_layer(indim, outdim))
	return network(layers, acvfn, deriv_acvfn=deriv_acvfn, loss=loss, deriv_loss=deriv_loss, linear_on_final=linear_on_final, momentum_coef=momentum_coef)

class network:
	def __init__(self, layers: list[layer], acvfn, deriv_acvfn=None, loss=None, deriv_loss=None, linear_on_final = False, momentum_coef = None):
		# array of layers, activation, loss functions and their derivatives
		# dloss is an array of partial derivatives corresponding to the output layer
		self.acvfn = acvfn
		self.deriv_acvfn = deriv_acvfn
		self.deriv_loss = deriv_loss
		# self.deracvfn = deracvfn
		# self.dloss = dloss
		self.loss = loss
		self.layers = layers
		self.linear_on_final = linear_on_final
		self.momentum_coef = momentum_coef
		if self.momentum_coef:
			self.last_grad_change = [np.zeros(l.weights.shape) for l in layers]
		for l in layers[0:len(layers)]:
			l.acvfn = acvfn
			l.deriv_acvfn = deriv_acvfn
		if not self.linear_on_final:
			layers[-1].acvfn = np.vectorize(lambda x:x)
			layers[-1].deriv_acvfn = np.vectorize(lambda x:1)
		# self.grad_weights

	def sgd(self,xbatch, ybatch, stepsize):
		grad_arrays = [self.grad_array2(x,y) for x,y in zip(xbatch,ybatch)]
		master_array = grad_arrays[0]
		# print(master_array)
		for array in grad_arrays[1::]:
			master_array = [master_array[i] + array[i] for i in range(len(array))]
		# print("master array:")
		# print(master_array)
		# print(grad_arrays[0])
		# print(grad_arrays[1])
		# print("-----------")
		for i in range(len(self.layers)):
			# print(master_array[i])
			# print(self.layers[i].weights)
			if self.momentum_coef:
				# print(master_array[i])
				self.layers[i].weights = self.layers[i].weights - ((1-self.momentum_coef)*(stepsize / len(xbatch))) * master_array[i] + (self.momentum_coef) * self.last_grad_change[i]
			else:
				self.layers[i].weights = self.layers[i].weights - (stepsize/len(xbatch)) * master_array[i]
			# print(self.layers[i].weights)
		self.last_grad_change = [-(stepsize / len(xbatch)) * master_array[i] + (self.momentum_coef) * self.last_grad_change[i] for i in range(len(self.layers))]

	def grad_array2(self,x,y):
		predicted_y = self.compute(x)
		L = len(self.layers)
		out_array = [np.zeros([1])] * L
		delta_array = [np.zeros([1])] * L
		# first output layer
		# print(L-1)
		delta_array[L-1] = np.multiply(self.layers[L-1].deriv_acvfn(self.layers[L-1].activation), self.deriv_loss(predicted_y, y))
		out_array[L-1] = np.matmul(delta_array[L-1],self.layers[L-2].out.transpose())
		# here l runs from L-2 to 1
		for l in range(L-2,0, -1):
			# print(out_array)
			# print(l)
			delta_array[l] = np.multiply(self.layers[l].deriv_acvfn(self.layers[l].activation),np.matmul(self.layers[l+1].weights.transpose(), delta_array[l+1]))
			# print(delta_array[l])
			out_array[l] = np.matmul(delta_array[l], self.layers[l-1].out.transpose())
		# delta_array[0] = self.deriv_acvfn(np.matmul(self.layers[1].weights.transpose(), delta_array[1]))
		delta_array[0] = np.multiply(self.layers[0].deriv_acvfn(self.layers[0].activation),np.matmul(self.layers[1].weights.transpose(), delta_array[1]))
		out_array[0] = np.matmul(delta_array[0], x.transpose())
		# print(out_array)
		# print()
		return out_array

	# def dump_to_json(self):


	def compute(self,x):
		# push results forward
		# print(f"x: {x}")
		for l in self.layers:
			x = l.compute(x)
		return x

if __name__ == "__main__":
	# sigmoid
	# active = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
	# dactive =np.vectorize(lambda x: (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x)))))
	# leaky relu
	p=0.1
	active = np.vectorize(lambda x: max(p*x,x))
	dactive = np.vectorize(lambda x: np.piecewise(x,[x>=0,x<0],[1,p]))
	loss = lambda y1,y2: 0.5*np.dot(y1-y2,y1-y2)
	dloss = lambda y1,y2: (y1-y2)

	def emp_risk(preddata, actualdata):
		return (1/len(preddata))*sum([loss(x,y) for x,y in zip(preddata,actualdata)])

	# l1 = layer(np.array([[1]]))
	# l2 = layer(np.array([[5,0], [6,3]]))
	# l3 = layer(np.array([[1]]))
	# net = network([l1,l3], active, dactive, loss, dloss)
	net = init_random_network([1,10,1],active, dactive, loss, dloss)
	# # total number of training data points
	# n = 100
	# # batch size
	# m = 10
	# xdata = np.linspace(0, np.pi, num=n)
	# for k in range(20):
	# 	np.random.shuffle(xdata)
	# 	traindata = np.cos(xdata)
	# 	xbatches = np.split(xdata, m)
	# 	ybatches = np.split(traindata, m)
	# 	for i in range(len(xbatches)):
	# 		preddata = [net.compute(np.array([[x]])) for x in xdata]
	# 		print(f"global risk at iter {k}, batch {i}: {emp_risk(preddata,traindata)}")
	# 		net.sgd([np.array([[x]]) for x in xbatches[i]],[np.array([[x]]) for x in xbatches[i]],0.001)
