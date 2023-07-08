from thinktank import init_random_network, layer
import numpy as np
# approaching xor gate from a random initialised sigmoid nnetwork via sgd.

active = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
dactive =np.vectorize(lambda x: (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x)))))
sftmax = lambda x:(1/np.sum(np.exp(x)))*np.exp(x)
# gives jacobian
def dsftmax(x):
    # x should be a col array
    jacob = np.zeros((x.shape[0], x.shape[0]))
    id = np.identity(x.shape[0])
    sf = sftmax(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            jacob[j,i] = sf[i] * (id[i,j] - sf[j])
    return jacob
# loss = lambda y1,y2: 0.5*np.dot(y1-y2,y1-y2)
# dloss = lambda y1,y2: (y1-y2)
loss = lambda y1,y2: (-1/y1.shape[0])*(np.multiply(y2,np.log(y1)).sum())
dloss = lambda y1,y2: (-1/y1.shape[0])*(np.multiply(y2,np.reciprocal(y1)))

net = init_random_network([2,5,2], active, dactive, loss, dloss,momentum_coef=9/10)
net.layers[-1].acvfn = sftmax
net.layers[-1].deriv_acvfn = dsftmax
input = [np.array([x]).transpose() for x in [[1,1],[0,1],[1,0],[0,0]]]
output = [np.array([x]).transpose() for x in [[0,1],[1,0],[1,0],[0,1]]]

def emp_risk(preddata, actualdata):
    # print([(x, y) for x, y in zip(preddata, actualdata)])
    return (1 / len(preddata)) * sum([loss(x, y) for x, y in zip(preddata, actualdata)])

# print(emp_risk([np.array([x]) for x in [1,0.5,0,0]], output))

# exit()
print("initial predictions and risk:")
print(f"risk: {emp_risk([net.compute(x) for x in input], output)}")
print("input_pair xor_output net_output")
for i, o in zip(input, output):
    print(f"({i[0,0]}, {i[1,0]})     {o[0,0]}          {net.compute(np.array([i]).transpose()).transpose()}")

data = list(zip(input,output))
# net.compute(list(data)[0][0])
for j in range(10000):
    for d in data:
        net.sgd([d[0]],[d[1]],0.05)
        # print(dloss(net.layers[-1].out, d[1]))
    # net.sgd(input,output,0.1)
    print(f"risk after round {j+1}: {emp_risk([net.compute(x) for x in input], output)}")


print("final predictions and risk:")
print(f"risk: {emp_risk([net.compute(x) for x in input], output)}")
print("input_pair xor_output net_output")
for i, o in zip(input, output):
    print(f"({i[0,0]}, {i[1,0]})     {o[0]}          {net.compute(np.array([i]).transpose())}")
