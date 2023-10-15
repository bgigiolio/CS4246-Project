import csv
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
import tqdm
import random

#TODO - relate the danger data to the amount of ships through the area
#TODO - diagonal movement for the boats?
#TODO - ints instead of floats? liley better to do when generating the mdp

def euclidean(t1:tuple, t2:tuple):
    return math.dist(t1,t2)

class Dataset:
    "note that the self.states attribute currently {(lon,lat):{neighbours:list, n_attacks:int, danger:float, to_goal:float}}, but will be updated continously. \
    the neighbours list contains (state, index) where index 0:right, 1:up, 2:left, 3:down. Up means an increase in latitude and right means an increase in longitude.\
    Increase in latitude mean going north, increase in longitude means going east. -90<=lat<=90 degrees.\
    Due to float rounding issues it was hardcoded so that the state is represented as a tuple of floats with 4 decimal digits."
    
    def __init__(self, min_lon:float, max_lon:float, min_lat:float, max_lat:float):
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.total_danger = None
        self.total_number_of_attacks = None
        self.goal=None
        self.start=None
        self.states = None

    def __str__(self):
        if self.states:
            state, attributes = random.choice(list(self.states.items()))
        else:
            state, attributes = None, None
        return f"Dataset with metaparameters: (self.min_lon={self.min_lon}, self.max_lon={self.max_lon}, self.min_lat={self.min_lat}, self.max_lat={self.max_lat}), \
            total danger={self.total_danger}, total number of attacks={self.total_number_of_attacks}, goal={self.goal}, start={self.start}.\
            A random state with attributes in the dataset: {state, attributes})"

    def set_start_goal_generate_distance(self, start:tuple, goal:tuple):
        "set start goal and adds or replaces the distance_to_goal metric of every state. \
        If the start and/or goal argument is not a state maps the given tuple to closest state generated."

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
    dataset.generate_states(distance=1) #needs to be done first
    dataset.load_pirate_data(spread_of_danger=1)
    dataset.set_start_goal_generate_distance(start=(-10, 91), goal=(24, 140))
    print(dataset) #this shows a random example state

if __name__=='__main__':
    main()