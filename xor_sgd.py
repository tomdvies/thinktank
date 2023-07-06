from thinktank import init_random_network, layer
import numpy as np
# approaching xor gate from a random initialised sigmoid nnetwork via sgd.

active = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
dactive =np.vectorize(lambda x: (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x)))))
# loss = lambda y1,y2: 0.5*np.dot(y1-y2,y1-y2)
# dloss = lambda y1,y2: (y1-y2)
loss = lambda y1,y2: -y2*np.log(y1) - (1-y2)*np.log(1-y1)
dloss = lambda y1,y2: -y2/y1 + (1-y2)/(1-y1)
net = init_random_network([2,5,1], active, dactive, loss, dloss, activate_on_final=True)

input = [np.array([x]).transpose() for x in [[1,1],[0,1],[1,0],[0,0]]]
output = [np.array([x]) for x in [0,1,1,0]]

def emp_risk(preddata, actualdata):
    # print([(x, y) for x, y in zip(preddata, actualdata)])
    return (1 / len(preddata)) * sum([loss(x, y) for x, y in zip(preddata, actualdata)])

# print(emp_risk([np.array([x]) for x in [1,0.5,0,0]], output))

# exit()
print("initial predictions and risk:")
print(f"risk: {emp_risk([net.compute(x) for x in input], output)[0,0]}")
print("input_pair xor_output net_output")
for i, o in zip(input, output):
    print(f"({i[0,0]}, {i[1,0]})     {o[0]}          {net.compute(np.array([i]).transpose())[0,0,0]}")

data = list(zip(input,output))
# net.compute(list(data)[0][0])
for j in range(10000):
    for d in data:
        net.sgd([d[0]],[d[1]],0.1)
        # print(dloss(net.layers[-1].out, d[1]))
    # net.sgd(input,output,0.1)
    print(f"risk after round {j+1}: {emp_risk([net.compute(x) for x in input], output)[0,0]}")
# exit()
for j in range(10000):
    for d in data:
        net.sgd([d[0]],[d[1]],0.05)
    # net.sgd(input,output,0.01)
    print(f"risk after round {j+1}: {emp_risk([net.compute(x) for x in input], output)[0,0]}")

print("final predictions and risk:")
print(f"risk: {emp_risk([net.compute(x) for x in input], output)}")
print("input_pair xor_output net_output")
for i, o in zip(input, output):
    print(f"({i[0,0]}, {i[1,0]})     {o[0]}          {net.compute(np.array([i]).transpose())[0,0][0]}")

for l in net.layers:
    print(l.weights)