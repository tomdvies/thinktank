import numpy as np
import base64

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
		# apply weights and store for backprop
		self.activation = np.matmul(self.weights, x)
		# add bias
		# x2 = x1 + self.bias
		# apply activation function
		self.out = self.acvfn(self.activation)
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
		# activation function to give to layers that don't already have one
		self.acvfn = acvfn
		# either standard derivative of your activation function if its R->R or it's jacobian if it's R^n->R^n
		self.deriv_acvfn = deriv_acvfn
		# grad of your loss function with respect to the output of your nn
		self.deriv_loss = deriv_loss
		# the loss function to be used in sgd
		self.loss = loss
		self.layers = layers
		self.momentum_coef = momentum_coef
		if self.momentum_coef:
			self.last_grad_change = [np.zeros(l.weights.shape) for l in layers]
		self.last_grad_change_set = False
		for l in layers[0:len(layers)]:
			# if the layers don't currently have activation functions, add them
			if not l.acvfn:
				l.acvfn = acvfn
				l.deriv_acvfn = deriv_acvfn
		# hacky for if you want the last layer to have an activation x
		if linear_on_final:
			layers[-1].acvfn = np.vectorize(lambda x:x)
			layers[-1].deriv_acvfn = np.vectorize(lambda x:1)
		# self.grad_weights

	def multiply(self,A,B):
		# hacky fix to avoid a big re-write while still allowing for activation functions from R^n -> R^n instead of just R -> R
		# in the 1st case layer.deriv_acfn should be the jacobian of your activation function
		# to do properly, should enforce that layer.deriv_acfn is the diagonal jacobian of the vectorised version of your 1d activation function.
		if A.shape[1] == 1:
			return np.multiply(A,B)
		else:
			return np.matmul(A,B)

	def sgd(self,xbatch, ybatch, stepsize):
		# batches work as the expectation of them is still grad of empirical risk on your entire training set
		# assuming they are chosen uniformly from your train data
		# get arrays of derivatives
		grad_arrays = [self.grad_array(x,y) for x,y in zip(xbatch,ybatch)]
		# construct sum of derivatives
		master_array = grad_arrays[0]
		for array in grad_arrays[1::]:
			master_array = [master_array[i] + array[i] for i in range(len(array))]
		for i in range(len(self.layers)):
			# either do normal sgd or sgd with momentum
			if self.last_grad_change_set and self.momentum_coef:
				# print(master_array[i])
				# print(self.last_grad_change)
				self.layers[i].weights = self.layers[i].weights - ((1-self.momentum_coef)*(stepsize / len(xbatch))) * master_array[i] + (self.momentum_coef) * self.last_grad_change[i]
			else:
				self.layers[i].weights = self.layers[i].weights - (stepsize/len(xbatch)) * master_array[i]
		if self.momentum_coef:
			# update last grad change for momentum
			self.last_grad_change = [- ((1-self.momentum_coef)*(stepsize / len(xbatch))) * master_array[i] + (self.momentum_coef) * self.last_grad_change[i] for i in range(len(self.layers))]
			self.last_grad_change_set = True


	def grad_array(self,x,y):
		predicted_y = self.compute(x)
		L = len(self.layers)
		out_array = [np.zeros([1])] * L
		delta_array = [np.zeros([1])] * L
		# output layer handled as it's own case
		delta_array[L-1] = self.multiply(self.layers[L-1].deriv_acvfn(self.layers[L-1].activation), self.deriv_loss(predicted_y, y))
		out_array[L-1] = np.matmul(delta_array[L-1],self.layers[L-2].out.transpose())
		# loop over inside layers and do backprop
		# here l runs from L-2 to 1
		for l in range(L-2,0, -1):
			delta_array[l] = self.multiply(self.layers[l].deriv_acvfn(self.layers[l].activation),np.matmul(self.layers[l+1].weights.transpose(), delta_array[l+1]))
			out_array[l] = np.matmul(delta_array[l], self.layers[l-1].out.transpose())
		# input layer handled as it's own case
		delta_array[0] = self.multiply(self.layers[0].deriv_acvfn(self.layers[0].activation),np.matmul(self.layers[1].weights.transpose(), delta_array[1]))
		out_array[0] = np.matmul(delta_array[0], x.transpose())
		return out_array

	def compute(self,x):
		# push results forward through each layer
		for l in self.layers:
			x = l.compute(x)
		return x

	def dump_network_to_string(self):
		# saves network weights to a string, base64 encoding each weight matrix.
		return ";".join([base64.b64encode(l.weights.astype("<u2").tobytes()).decode() for l in self.layers])

	def load_network_from_string(self, s: str):
		# load weights from string generated via the above function.
		weights  = [np.frombuffer(base64.b64decode(subs)) for subs in s.split(";")]
		for l,w in zip(self.layers, weights):
			l.weights = w.reshape(l.weights.shape)

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
