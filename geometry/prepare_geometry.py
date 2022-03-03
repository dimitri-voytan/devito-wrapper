'''
An example of how to prepare the acquisition geometry.
'''
import pickle
import numpy as np

dx = 6.096  # m
dz = 6.096  # m

# Setup source
source_x = np.linspace(0, 3617*dx, 20)
source_z = dx*3

recv_x = np.linspace(0, 3617*dx, 3617)
recv_z = dz*3

try:
    n_sources = max(len(source_x), len(source_x))
except TypeError:
    # For a single source float type will has no len()
    n_sources = max(len([source_x]), len([source_z]))

# setup source coordinates
sources = np.empty((n_sources, 2))
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
