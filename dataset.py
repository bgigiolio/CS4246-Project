import csv
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
import tqdm
import random
import json
import os

#TODO - relate the danger data to the amount of ships through the area
#TODO - diagonal movement for the boats?

def read_dataset(name):
    "returns a stored dataset object. The name should include not inlcude .json and the dataset should be in the saved_datasets map"
    file_path=f"saved_datasets/{name}.json"
    with open(file_path, 'r') as f:
        dataset = json.load(f)

    states_str=dataset['states']
    states={}
    for key in tqdm.tqdm(states_str.keys(), 'reading saved dataset'):
        floats=key[1:-2].split(',')
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

        if not os.path.exists(file_path) or erase_current_content:
            with open(file_path, 'w') as f:
                f.write('')

        with open(file_path, 'w') as f:
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

def main():
    "an example on how to generate a complete dataset using this code"
    dataset=Dataset(88.6, 152.9, -12.4, 31.3) #South East Asia
    scale = 1
    dataset.generate_states(distance=scale) #needs to be done first
    dataset.load_pirate_data(spread_of_danger=1)
    dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
    print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
    dataset.save("dataset_1")
    dataset=read_dataset("dataset_1")
    print(dataset.states[dataset.get_closest_state(90, 0)]['neighbours']) #this is working as intended

    #plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)

if __name__=='__main__':
    main()