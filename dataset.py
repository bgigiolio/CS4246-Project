import csv
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
import tqdm
import random
import json
import os
import rasterio
from visualize_dataset import plot_dataset_on_map

# not important
#TODO - diagonal movement for the boats?
#TODO - information about nearby countries in order to use for Deep learning solution

#NOT:
# - normalise density and reward, can easily be done by looping over the keys if that is needed when making the reward function

### SUPPLEMENTARY FUNCTIONS ###

def get_index(input:float, lon_input:bool):
    "The transformation used by the dataset is 0.005, 0.0, -180.015311275, 0.0, -0.005, 85.00264793700009"
    if lon_input:
        return int((input+180.015311275)/0.005)
    else:
        return int((85.00264793700009-input)/0.005)
    
def get_lon_lat(row,col):
    return 0.005*col-180.015311275,85.00264793700009-0.005*row #row is latituide
    
def local_average(dataset, row, colon, raster_array, window_size=200, method:str="local_averege"):
    "returns the local averege of the density close to that region\
        chose either local_averege, local_max or local_sea_averege (untested+slow)"
    rows, cols=np.shape(raster_array)
    row_min=max(int(row-window_size/2),0) #
    row_max=min(int(row+window_size/2),rows-1)
    col_min=max(int(colon-window_size/2),0)
    col_max=min(int(colon+window_size/2),cols-1)

    if method=="local_averege":
        return np.mean(raster_array[row_min:row_max, col_min:col_max]) #the averege locally, maybe not great because port of Shanghai effect
    elif method=="local_max":
        return np.max(raster_array[row_min:row_max, col_min:col_max]) #then the coast becomes interesting again
    else: #need to check what very dense local points is land -> about 40 000 times slower..., too inpractical?
        buffer=2
        m = Basemap(llcrnrlon=dataset.min_lon-buffer,
                llcrnrlat=dataset.min_lat-buffer,
                urcrnrlon=dataset.max_lon+buffer,
                urcrnrlat=dataset.max_lat+buffer,
                lat_0=(dataset.max_lat - dataset.min_lat)/2,
                lon_0=(dataset.max_lon-dataset.min_lon)/2,
                projection='merc',
                resolution = 'h', #high resolution
                area_thresh=10000.,
                )
        
        lons=[]
        lats=[]
        for row in range(row_min, row_max+1):
            for col in range(col_min, col_max+1):
                lon,lat=get_lon_lat(row,col)
                lons.append(lon)
                lats.append(lat)
        
        lons_m, lats_m=m(np.array(lons), np.array(lats))

        #A potential speed up could be using Monte Carlo sampling...
        #ro, co=np.shape(raster_array[row_min:row_max, col_min:col_max])
        #total=ro*co
        
        sea_points=0
        for lon_m in lons_m:
            for lat_m in lats_m:
                print(int(not m.is_land(lon_m, lat_m)))
                sea_points+=int(not m.is_land(lon_m, lat_m))

        return np.sum(raster_array[row_min:row_max, col_min:col_max])/sea_points
        


def read_dataset(name):
    "returns a stored dataset object. The name should include not inlcude .json and the dataset should be in the saved_datasets map"
    file_path=f"saved_datasets/{name}.json"
    with open(file_path, 'r') as f:
        dataset = json.load(f)

    #translating goal and start to touples instead of list
    [lon,lat]=dataset['start']
    dataset['start']=(lon,lat)
    [lon,lat]=dataset['goal']
    dataset['goal']=(lon,lat)

    states_str=dataset['states']
    states={}
    for key in tqdm.tqdm(states_str.keys(), 'reading saved dataset'):
        floats=key[1:-1].split(',')
        key_tuple=(float(floats[0]), float(floats[1]))
        states[key_tuple]=states_str[key]
        #changing from list of lists to touples because json don't like tuples
        neighbour_list=states[key_tuple]['neighbours']
        for i in range(len(neighbour_list)):
            [[lon, lat], index]=neighbour_list[i]
            neighbour_list[i]=((lon,lat), index)
        states[key_tuple]['neighbours']=neighbour_list

    return Dataset(dataset['min_lon'], dataset['max_lon'], dataset['min_lat'], dataset['max_lat'], \
                   dataset['total_danger'], dataset['total_number_of_attacks'], dataset['goal'], dataset['start'], states, dataset['scale'])

def euclidean(t1:tuple, t2:tuple):
    return math.dist(t1,t2)


### The class ###

class Dataset:
    "note that the self.states attribute currently {(lon,lat):{neighbours:list, n_attacks:int, danger:float, to_goal:float}}, but will be updated continously. \
    the neighbours list contains (state, index) where index 0:right, 1:up, 2:left, 3:down. Up means an increase in latitude and right means an increase in longitude.\
    Increase in latitude mean going north, increase in longitude means going east. -90<=lat<=90 degrees.\
    Due to float rounding issues it was hardcoded so that the state is represented as a tuple of floats with 4 decimal digits."
    
    def __init__(self, min_lon:float, max_lon:float, min_lat:float, max_lat:float, total_danger=None, total_number_of_attacks=None, goal=None, start=None, states:dict=None, scale=None):
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.total_danger = total_danger
        self.total_number_of_attacks = total_number_of_attacks 
        self.goal=goal
        self.start=start
        self.states = states
        self.scale=scale

    def __str__(self):
        if self.states:
            state, attributes = random.choice(list(self.states.items()))
        else:
            state, attributes = None, None
        return f"Dataset with metaparameters: (self.min_lon={self.min_lon}, self.max_lon={self.max_lon}, self.min_lat={self.min_lat}, self.max_lat={self.max_lat}), \
            total danger={self.total_danger}, total number of attacks={self.total_number_of_attacks}, goal={self.goal}, start={self.start}.\
            A random state with attributes in the dataset: {state, attributes})"
    
    def get_closest_state(self, lon_query, lat_query):
        "returns the closest state (lon,lat) in the dataset which is closest to the query" 
        state=min(self.states.keys(), key=lambda k: euclidean((lon_query,lat_query), k))
        return state

    def save(self, name=None, erase_current_content=True):
        "Saves the dataset in the saved_datasets map. The format is a dictionary with each attribute a seperate key value pair. The tuple keys are changed to strings because of json.\
        Deafult name is self.__str__ stripped of spaces. No need to add .json manually. Overwrites potential current content in file unless erase_current_content=False.\
        Must have states generated to be saved."
        
        #json files cannot have float keys -> changing all tuples to strings and then back
        states_str={}
        states=self.states
        for key in tqdm.tqdm(states.keys(), 'saving dataset'):
            states_str[str(key)]=states[key]
        
        dataset_dict = {
        "min_lon": self.min_lon,
        "max_lon": self.max_lon,
        "min_lat": self.min_lat,
        "max_lat": self.max_lat,
        "total_danger": self.total_danger,
        "total_number_of_attacks": self.total_number_of_attacks,
        "goal": self.goal,
        "start": self.start,
        "states": states_str,
        "scale":self.scale
        }
        if not name:
            name = self.__str__().strip().replace(" ", "")
        file_path=f"saved_datasets/{name}.json"

        # create dir if does not exist
        try:  
            os.makedirs("saved_datasets/", exist_ok=True)  
        except OSError as error: 
            print(error)

        if not os.path.exists(file_path) or erase_current_content:
            with open(file_path, 'w+') as f:
                f.write('')

        with open(file_path, 'w+') as f:
            json.dump(dataset_dict, f)

    def set_start_goal_generate_distance(self, start:tuple, goal:tuple):
        "set start goal and adds or replaces the distance_to_goal metric of every state. \
        If the start and/or goal argument is not a state, maps the given tuple to closest state generated."

        states=self.states
        if not states:
            print("Need to generate the states first!")
            raise ValueError
        
        lon_start, lat_start=min(states.keys(), key=lambda k: euclidean(start, k))
        lon_goal, lat_goal=min(states.keys(), key=lambda k: euclidean(goal, k))
        self.goal=(lon_goal, lat_goal)
        self.start=(lon_start, lat_start)

        for state in tqdm.tqdm(states.keys(), desc="calculating distance to goal"):
            states[state]["to_goal"]=euclidean(state,goal)

        self.states=states

    def generate_states(self, distance:float=1):
        "generates (lon, lat) by equal spacing over the entire region and deleting everything which is not ocean.\
        Updates max lon and lat so that the corners of the area given by (self.max_lon, self.min_lon, self.max_lat, self.min_lat) are states (unless they are on land).\
        Due to float rounding issues it was hardcoded so that the state is represented as a tuple of floats with 4 decimal digits."

        states={}
        if self.states:
            print("states allready generated")
            return False

        self.scale=distance

        # create map using BASEMAP
        buffer=2
        m = Basemap(llcrnrlon=self.min_lon-buffer,
                llcrnrlat=self.min_lat-buffer,
                urcrnrlon=self.max_lon+buffer,
                urcrnrlat=self.max_lat+buffer,
                lat_0=(self.max_lat - self.min_lat)/2,
                lon_0=(self.max_lon-self.min_lon)/2,
                projection='merc',
                resolution = 'h', #high resolution
                area_thresh=10000.,
                )

        i_lon_max=math.ceil((self.max_lon-self.min_lon)/distance)
        i_lat_max=math.ceil((self.max_lat-self.min_lat)/distance)
        self.max_lon = self.min_lon+(i_lon_max-1)*distance 
        self.max_lat = self.min_lat+(i_lat_max-1)*distance
        for i_lon in tqdm.tqdm(range(i_lon_max), desc="generating states"):
            for i_lat in range(i_lat_max):
                lon=self.min_lon+i_lon*distance
                lat=self.min_lat+i_lat*distance
                lon_m, lat_m=m(np.array(lon), np.array(lat)) #convert to map cordinates so that basemap can check for land
                if not m.is_land(lon_m, lat_m):
                    state_attributes={}
                    states[(round(self.min_lon+i_lon*distance,4), round(self.min_lat+i_lat*distance,4))]=state_attributes
                    neighbours=[]
                    for k in [(0,(1,0)), (1,(0,1)), (2,(-1,0)), (3,(0,-1))]: 
                        if self.min_lon<=lon+k[1][0]*distance<self.min_lon+i_lon_max*distance and \
                            self.min_lat<=lat+k[1][1]*distance<self.min_lat+i_lat_max*distance:
                            lon_m, lat_m=m(np.array(lon+k[1][0]*distance),np.array(lat+k[1][1]*distance))
                            if not m.is_land(lon_m, lat_m):
                                neighbours.append(((round(lon+k[1][0]*distance,4), round(lat+k[1][1]*distance,4)),k[0]))
                        
                    state_attributes["neighbours"]=neighbours

        self.states=states

    def load_pirate_data(self, spread_of_danger:int=1, path:str='pirate_attacks.csv', danger_ranking:dict={"Fired Upon":15, "Hijacked":10, "Explosion":7, "Boarding":5, "Boarded":5, "Attempted":1, "Suspicious":0.5, "NA":1}, gamma=0.5):
        "Adds the total danger and total atacks data to the entire dataset. Adds the danger, n_attacks attribute to the state dictionary. \
        The danger_ranking is a subjective option about the severity of different types of attacks. gamma is the discount factor which the danger from an attacks is spread by."
        
        min_lon=self.min_lon
        max_lon=self.max_lon
        min_lat=self.min_lat
        max_lat=self.max_lat
        total_danger=0
        number_of_attacks=0
        states=self.states

        if not states:
            print("Generate the states before loading the data!")
            raise ValueError

        #initialasing every state with a value for attacks:
        for state in states.keys():
            states[state]["n_attacks"]=0
            states[state]["danger"]=0

        with open(path, 'r', encoding="utf8") as file:
            reader = csv.reader(file)
            i=0
            for row in reader:
                if i:
                    lon=float(row[2])
                    lat=float(row[3])
                    if min_lon<=lon<=max_lon and min_lat<=lat<=max_lat: #filtering the data to be inside the chosen region
                        if row[4] in danger_ranking.keys():
                            sample_danger=danger_ranking[row[4]]
                        else:
                            print(f"Danger not accounted for {row[4]}")
                            raise ValueError

                        total_danger+=sample_danger
                        number_of_attacks+=1
                        lon_state, lat_state=min(states.keys(), key=lambda k: euclidean((lon,lat), k))
                        states[(lon_state,lat_state)]["n_attacks"]+=1
                        states[(lon_state,lat_state)]["danger"]+=sample_danger
                        if spread_of_danger:
                            for neighbour in states[(lon_state,lat_state)]["neighbours"]:
                                #states[neighbour[0]]["n_attacks"]+=gamma*1 - Reasonable to keep this an int and hence not spread to neighbours
                                states[neighbour[0]]["danger"]+=gamma*sample_danger
                i=1
        self.states=states
        self.total_danger=total_danger
        self.total_number_of_attacks=number_of_attacks

    def add_trafic_density(self, method:str="local_averege"):
        "chose either local_max, local_averege, local_sea_averege (obs, very slow! not tested therefore)"
        file_path = 'ShipDensity_Commercial1.tif'
        raster_data = rasterio.open(file_path) 
        print("reading the huge shipping density file...")   
        raster_array = raster_data.read(1)
        states=self.states
        for (lon,lat) in tqdm.tqdm(states, desc="adding traffic density"):
            colon_index=get_index(lon, lon_input=True) #least latitude leads to the highest row index in the density dataset
            row_index=get_index(lat, lon_input=False)
            density=local_average(self, row_index,colon_index, raster_array, int(self.scale/(0.005)), method)
            states[(lon,lat)]['density']=density
        self.states=states

def main():
    "an example on how to generate a complete dataset using this code"
    if True:
        dataset=Dataset(88.6, 152.9, -12.4, 31.3) #South East Asia
        scale = 1
        dataset.generate_states(distance=scale) #needs to be done first
        dataset.load_pirate_data(spread_of_danger=1)
        dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
        dataset.add_trafic_density(method="local_averege") 
        print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
        #plot_dataset_on_map(dataset, Attribute="density", Ranges=5)
        #dataset.save("dataset_demo")
    #dataset=read_dataset("dataset_demo")
    #print(dataset) 

    plot_dataset_on_map(dataset, Attribute="density", Ranges=5) 

if __name__=='__main__':
    main()