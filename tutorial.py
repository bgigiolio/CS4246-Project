### basemap ###

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap()

map.drawcoastlines()

plt.show()
plt.savefig('figures/test.png') #don't work

### built in mdp package - mdptoolbox ###

import mdptoolbox.example

P, R = mdptoolbox.example.forest(S=3, r1=4, r2=2, p=0.1, is_sparse=False)
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
vi.run()
print(vi.policy) # result is (0, 0, 0)

### gym and gymnasium ###

import gym
import gymnasium 
print(gym)
print(gymnasium)
