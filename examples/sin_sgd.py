from thinktank import init_random_network, layer
import numpy as np

# approaching cos from a random initialised sigmoid nnetwork via sgd.
active = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
dactive =np.vectorize(lambda x: (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x)))))
loss = lambda y1,y2: 0.5*np.dot(y1-y2,y1-y2)
dloss = lambda y1,y2: (y1-y2)
# loss = lambda y1,y2: -y2*np.log(y1) - (1-y2)*np.log(1-y1)
# dloss = lambda y1,y2: -y2/y1 + (1-y2)/(1-y1)
net = init_random_network([1,30,20,1], active, dactive, loss, dloss, activate_on_final=False,momentum_coef=3/10)

# total number of training data points
n = 1000
# batch size
m = 5
input = [np.array([[x]]).transpose() for x in np.linspace(0, 2*np.pi, num=n)]
np.random.shuffle(input)
output = [np.sin(x) for x in input]
batchinputs = [input[i:i + m] for i in range(0, len(input), m)]
batchoutputs = [output[i:i + m] for i in range(0, len(input), m)]
def emp_risk(preddata, actualdata):
    # print([(x, y) for x, y in zip(preddata, actualdata)])
    return (1 / len(preddata)) * sum([loss(x, y) for x, y in zip(preddata, actualdata)])

# print(emp_risk([np.array([x]) for x in [1,0.5,0,0]], output))


print(f"initial risk: {emp_risk([net.compute(x) for x in input], output)}")
# print(batchinputs)
data = list(zip(batchinputs,batchoutputs))

for j in range(50):
    for d in data:
        # print(d[1])
        net.sgd(d[0],d[1],0.4)
        # print(dloss(net.layers[-1].out, d[1]))
    # net.sgd(input,output,0.1)
    print(f"risk after round {j+1}: {emp_risk([net.compute(x) for x in input], output)[0,0]}")


for j in range(20):
    for d in data:
        net.sgd(d[0],d[1],0.2)
    # net.sgd(input,output,0.01)
    print(f"risk after round {j+1}: {emp_risk([net.compute(x) for x in input], output)[0,0]}")

for j in range(30):
    for d in data:
        net.sgd(d[0],d[1],0.1)
    # net.sgd(input,output,0.01)
    print(f"risk after round {j+1}: {emp_risk([net.compute(x) for x in input], output)[0,0]}")

print("final risk:")
print(f"risk: {emp_risk([net.compute(x) for x in input], output)[0,0]}")