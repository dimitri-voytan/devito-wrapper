'''
An example of how to prepare the acquisition geometry.
The output nav.pkl file should have a 
'''
import pickle
import numpy as np

dx = 20.0  # m
dz = 20.0  # m

#Setup source 
source_x = np.linspace(0, 500*dx, 20)
source_z = 20.0

recv_x = np.linspace(0, 500*dx, 400)
recv_z = 40.0

# setup source coordinates
sources = np.empty((len(source_x), 2))
sources[:, 0] = source_x
sources[:, 1] = source_z

nav = []

# setup reciever coordinates
for i in range(sources.shape[0]):
    source = sources[i]
    recievers = np.empty((len(recv_x), 2))
    recievers[:, 0] = recv_x
    recievers[:, 1] = recv_z

    nav.append([source, recievers])

with open('nav.pkl', 'wb') as f:
    pickle.dump(nav, f)
