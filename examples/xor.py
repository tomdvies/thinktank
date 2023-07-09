from thinktank import network, layer
import numpy as np
# xor gate implementation with relu as the activation function.
layers = []
layers.append(layer(np.array([[-1,1], [1,-1]])))
layers.append(layer(np.array([[1,1]])))
# relu
active = np.vectorize(lambda x: max(0, x))
net = network(layers,active)
net.load_network_from_string("AAAAAAAA8L8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8L8=;AAAAAAAA8D8AAAAAAADwPw==")
input = [[1,1],[0,1],[1,0],[0,0]]
output = [0,1,1,0]
print("input_pair xor_output net_output")
for i, o in zip(input, output):
    print(f"({i[0]}, {i[1]})     {o}          {net.compute(np.array([i]).transpose())[0,0]}")
print(net.dump_network_to_string())