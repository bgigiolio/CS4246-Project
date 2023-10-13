import csv
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
import tqdm

#standard_danger={"Hijacked":10, "Boarded":5, "Attempted":1}

#DON't stand still, move forward, make things, break things

#brainstorming of suggestions on what to do next after discussion with team:
#TODO - merge all the classes into one class which handles the entire mdp problem, and you run certain methods on it to load datasets on it
#TODO - calculate reward per state needed, otherwise pretty much done, ready to be put into a MDP solver (and visualized
#TODO MDP start, goal
#TODO - make even nicer
#TODO - metadata object of the mdp?
#TODO - save in file (Blake is allready looking into this, using JSON), also save training data in file for future RL
#TODO - relate the danger data to the amount of ships through the area
#TODO - make a currently best version making us of our currently implemented ideas
#TODO - seperate into subfiles for example creation of mdp (this file) - visualisation - main - algoritms
#TODO - modellera states som graf? use those packages, initialize like that

def euclidean(t1:tuple, t2:tuple):
    (x1,y1)=t1
    (x2, y2)=t2
    return ((x1-x2)**2+(y1-y2)**2)**0.5

class MDP:
    def __init__(self, region,):
        self.metadata = region
        self.states = {}
        #self.actions = None # should be part of states {state:attributes_dict}
        #self.rewards = None 
        #self.transitions = None
        self.start_state = None #important, needed when generating the states? nej, vill kunna reuse
        self.goal_state = None

    def __str__(self):
        return f"MDP: "

    def set_start_goal(self, start:tuple, goal:tuple):
        "set start goal and adds or replaces the distance_to_goal metric of every state"
        states=self.states
        if start not in states.keys() or goal not in states.keys():
            print("start or stop not valid")
            return None #or move to the closest state?
        
        self.goal_state=goal
        self.start_state=start

        for state in states.keys():
            print("add distance_to_goal:euclidean(state, goal)")
            #do we actually need this? a little bit cheaty but should be useful, espcially for a functional approximation. can decide to not use it later


    def generate_states(self, distance=1, index=False):
        # Generate states based on the region and dataset
        pass

    def calculate_rewards(self):
        # Calculate rewards based on the dataset
        pass

    def calculate_transitions(self): #allready known?
        # Calculate transitions based on the states and actions
        pass 

    def solve(self):
        # Solve the MDP using a reinforcement learning algorithm
        pass

class Region:
    def __init__(self, min_lat, max_lat, min_lon, max_lon, name:str):
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.states = None
        self.name=name

    def __str__(self):
        return f"Region {self.name}: ({self.min_lat}, {self.max_lat}, {self.min_lon}, {self.max_lon})"
    
    def generate_states(self, distance:float=1, index=False):
        "generates (lon, lat) by equal spacing over the entire region and deleting everything which is not ocean\
        adds Region.states={state:{neighbours:[neigbour1, neighbour2]}}, a state is (lat,lon)\
        Todo if we want: make every state indexed by ints instead of (lon,lat), much becomes bad in that case though"

        #it is great to give distance because than we gaurentee that the travel time between every state is reasonably sort of the same

        #could be extended by adding weather conditions making certain transistions extra quick

        #could potentially be extended to moving all states which hits land and than calculate the travel time between states dynamically and use that in the reward function...
        
        #the possible actions also becomes very simple as you can simply look at the neighbours, if you sample tight enough they should not have land between each other

        states={}

        # create map using BASEMAP
        m = Basemap(llcrnrlon=self.min_lon,
                llcrnrlat=self.min_lat,
                urcrnrlon=self.max_lon,
                urcrnrlat=self.max_lat,
                lat_0=(self.max_lat - self.min_lat)/2,
                lon_0=(self.max_lon-self.min_lon)/2,
                projection='merc',
                resolution = 'h', #high resolution
                area_thresh=10000.,
                )

        i_lon_max=math.ceil((self.max_lon-self.min_lon)/distance)
        i_lat_max=math.ceil((self.max_lat-self.min_lat)/distance)
        for i_lon in tqdm.tqdm(range(i_lon_max), desc="generating states"):
            for i_lat in range(i_lat_max):
                lon=self.min_lon+i_lon*distance
                lat=self.min_lat+i_lat*distance
                lon_m, lat_m=m(np.array(lon), np.array(lat)) #convert to map cordinates so that basemap can check for land
                if not m.is_land(lon_m, lat_m):
                    state_attributes={}
                    states[(self.min_lon+i_lon*distance, self.min_lat+i_lat*distance)]=state_attributes
                    neighbours=[]
                    for k in [1, -1]:
                        if self.min_lon<=lon+k*distance<self.min_lon+i_lon_max*distance: #to make sure the neighbours are also states they have to be generated in the same way
                            lon_m, lat_m=m(np.array(lon+k*distance),np.array(lat)) #very slow unfortunatley...
                            if not m.is_land(lon_m, lat_m):
                                neighbours.append((lon+k*distance,lat))
                        if self.min_lat<=lat+k*distance<self.min_lat+i_lat_max*distance:
                            lon_m, lat_m=m(np.array(lon),np.array(lat+k*distance))
                            if not m.is_land(lon_m, lat_m):
                                neighbours.append((lon, lat+k*distance))
                    state_attributes["neighbours"]=neighbours
                    states[(lon,lat)]=state_attributes

        self.states=states

class Dataset:
    def __init__(self, min_lon, max_lon, min_lat, max_lat, total_danger, number_of_attacks, state_data):
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.total_danger = total_danger
        self.number_of_attacks = number_of_attacks
        self.states = state_data

    def __str__(self):
        return f"Dataset: ({self.min_lon}, {self.max_lon}, {self.min_lat}, {self.max_lat}, {self.total_danger}, {self.number_of_attacks}, {self.states})"

def read_data(region, spread_of_danger:int=1, path:str='pirate_attacks.csv', danger:dict={"Fired Upon":15, "Hijacked":10, "Explosion":7, "Boarding":5, "Boarded":5, "Attempted":1, "Suspicious":0.5, "NA":1}):
    "by deafult region south_east_asia={min_lat:-11.0, max_lat:25.0, min_lon:90.0, max_lon:145.0} WITH states generated\
        approximation to int which is divisible by approximate_to, if it is 0 no approximation,\
        returns dataset object containing, min_lon, max_lon, min_lat, max_lat, total_danger, total_number_of_attacks, states:dict\
        where states={state:{neighbours:list, danger:float, n_attacks:int}} (more attributes added continously)\
        Generalisable to other datasets quite easy"
    
    min_lon=region.min_lon
    max_lon=region.max_lon
    min_lat=region.min_lat
    max_lat=region.max_lat
    total_danger=0
    number_of_attacks=0
    states=region.states
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
                    if row[4] in danger.keys():
                        sample_danger=danger[row[4]]
                    else:
                        print(f"Danger not accounted for {row[4]}")
                        raise ValueError

                    total_danger+=sample_danger
                    number_of_attacks+=1
                    def distance(t1, t2):
                        x1,y1=t1
                        x2,y2=t2
                        return ((x2-x1)**2+(y2-y1)**2)**0.5
                    lon_state, lat_state=min(states.keys(), key=lambda k: distance((lon,lat), k))
                    states[(lon_state,lat_state)]["n_attacks"]+=1
                    states[(lon_state,lat_state)]["danger"]+=sample_danger
                    if spread_of_danger: #problemet med detta er att 
                        gamma=0.5
                        #print(states) (120,24) finns men inte (120,25) i states men i neighbours
                        for neighbour in states[(lon_state,lat_state)]["neighbours"]:
                            states[neighbour]["n_attacks"]+=gamma*1
                            states[neighbour]["danger"]+=gamma*sample_danger
            i=1
    
    dataset=Dataset(min_lon, max_lon, min_lat, max_lat, total_danger, number_of_attacks, states)
    return dataset

def plot_dataset_on_map(dataset, Attribute="danger", Ranges:int=5):
    "supply a state attributes which is a number such as danger or n_attacks, range 5 means 5 intervals as well as states with 0, regions according to (a,b]\
        a lot of of tedious data shuffling in this code"
    
    attribute_state_dict={}
    for state in tqdm.tqdm(dataset.states.keys(), desc="shuffling data"):
        if dataset.states[state][Attribute] in attribute_state_dict:
            attribute_state_dict[dataset.states[state][Attribute]].append(state)
        else:
            attribute_state_dict[dataset.states[state][Attribute]]=[state]

    #creating the range cutoffs:
    attributes=list(attribute_state_dict.keys())
    
    range_cuts=[i*max(attributes)/Ranges for i in range(Ranges+1)]

    #lost the number of samples of each attributes -> risk the ranges leads to most samples in the same range
    #attributes.sort() #
    #print(attributes)

    margin = 2 # buffer to add to the range
    lat_min = dataset.min_lat - margin
    lat_max = dataset.max_lat + margin
    lon_min = dataset.min_lon - margin
    lon_max = dataset.max_lon + margin

    # create map using BASEMAP
    m = Basemap(llcrnrlon=lon_min,
                llcrnrlat=lat_min,
                urcrnrlon=lon_max,
                urcrnrlat=lat_max,
                lat_0=(lat_max - lat_min)/2,
                lon_0=(lon_max-lon_min)/2,
                projection='merc',
                resolution = 'h', #high resolution
                area_thresh=10000.,
                )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color = 'white',lake_color='#46bcec') #color of oceans
    
    #same principle for all the 
    lons_0=[]
    lats_0=[]
    for j in range(len(attribute_state_dict[0])):
        lons_0.append(attribute_state_dict[0][j][0])
        lats_0.append(attribute_state_dict[0][j][1])
    lons_lats_by_range={(0,0):[lons_0, lats_0]} #format (a,b]

    for i in range(len(range_cuts)-1):
        #initialising every range
        lons_lats_by_range[(range_cuts[i], range_cuts[i+1])]=[[], []]

    for attribute in attribute_state_dict:
        for (a,b) in lons_lats_by_range.keys():
            if a<attribute<=b: #if the attribute in this range split all the states with that attribute into lons and lats lists and add to the range dict
                lons_lats_by_range[(a, b)][0]+=[attribute_state_dict[attribute][j][0] for j in range(len(attribute_state_dict[attribute]))]
                lons_lats_by_range[(a, b)][1]+=[attribute_state_dict[attribute][j][1] for j in range(len(attribute_state_dict[attribute]))]
    
    for (a,b) in lons_lats_by_range.keys():
        [lons, lats]=lons_lats_by_range[(a,b)]
        lons, lats = m(np.array(lons), np.array(lats)) #convert to map cordinates
        if a==b:
            m.scatter(lons, lats, marker = 'o', s=1, zorder=5, label=f'{Attribute} is {a}') #zorder makes the dots on top of the map
        else:
            m.scatter(lons, lats, marker = 'o', s=1, zorder=5, label=f'{a}<{Attribute}<={b}') #zorder makes the dots on top of the map
    plt.legend()
    plt.show()

if __name__=='__main__':
    south_east_asia=Region(-11, 25, 90, 145, "south_east_asia")
    south_east_asia.generate_states()
    dataset=read_data(south_east_asia)

    #print(dataset.states[(dataset.min_lat+1, dataset.min_lon+1)]) #indexes needed
    plot_dataset_on_map(dataset)


