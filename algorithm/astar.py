from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pprint import pprint

m = Basemap(projection="mill")
m.drawcoastlines()
arr = m.makegrid(10,10)
pprint(arr)