import csv
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math

#standard_danger={"Hijacked":10, "Boarded":5, "Attempted":1} #if not in scored as 1.5, hopefully only these 3 cases. More advanced analysis could probably be done lokking at the description of each attack 

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
    
    def generate_states(self, distance:float, index=False):
        "generates (lon, lat) by equal spacing over the entire region and deleting everything which is not ocean\
        adds Region.states={state:{neighbours:[neigbour1, neighbour2]}}, a state is (lat,lon)\
        Todo if we want: make every state indexed by ints instead of (lon,lat)"

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
        for i_lon in range(i_lon_max):
            for i_lat in range(i_lat_max):
                lon=self.min_lon+i_lon*distance
                lat=self.min_lat+i_lat*distance
                if not m.is_land(lon, lat):
                    state_attributes={}
                    states[(self.min_lon+i_lon*distance, self.min_lat+i_lat*distance)]=state_attributes
                    neighbours=[]
                    for k in [1, -1]:
                        if self.min_lon<lon<self.max_lon and self.min_lat<lat<self.max_lat:
                            if not m.is_land(lon+k*distance,lat):
                                neighbours.append((lon+k*distance,lat))
                            if not m.is_land(lon,lat+k*distance):
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
        self.state_data = state_data

    def __str__(self):
        return f"Dataset: ({self.min_lon}, {self.max_lon}, {self.min_lat}, {self.max_lat}, {self.total_danger}, {self.number_of_attacks}, {self.state_data})"

def round_to_divisible_by_a(Int, a):
    "tested, seems to be working"
    rest = Int % a
    if rest>a/2:
        Int=Int+(a-rest)
    else:
        Int=Int-rest
    return Int

def read_data(region=Region(-11, 25, 90, 145, "south_east_asia"), path:str='pirate_attacks.csv', approximate_to:int=1, danger:dict={"Hijacked":10, "Boarded":5, "Attempted":1}):
    "by deafult region south_east_asia={min_lat:-11.0, max_lat:25.0, min_lon:90.0, max_lon:145.0}, path pirate_attacks.csv, danger according to {Hijacked:10, Boarded:5, Attempted:1} 1.5 if not in dict\
        approximation to int which is divisible by approximate_to, if it is 0 no approximation,\
        returns dataset object containing, min_lon, max_lon, min_lat, max_lat, total_danger, total_number_of_attacks, states:dict\
        where states={state:{neighbours:list, danger:float, n_attacks:int}} (more attributes added continously)"
    
    min_lon=region.min_lon
    max_lon=region.max_lon
    min_lat=region.min_lat
    max_lat=region.max_lat
    total_danger=0
    number_of_attacks=0
    states_info={}
    with open(path, 'r', encoding="utf8") as file:
        reader = csv.reader(file)
        i=0
        for row in reader:
            if not i:
                print("header: "+str(row))
                i=1
            else:
                lon=float(row[2])
                lat=float(row[3])
                if min_lon<lon<max_lon and min_lat<lat<max_lat: #filtering the data to be inside the chosen region
                    if row[4] in danger.keys():
                        sample_danger=danger[row[4]]
                    else:
                        sample_danger=1.5
                    total_danger+=sample_danger
                    number_of_attacks+=1

                    if not region.states: 
                        if approximate_to:
                            lon=round_to_divisible_by_a(round(lon, 0), approximate_to)
                            lat=round_to_divisible_by_a(round(lat, 0), approximate_to)
                    else: 

                        print("here we want to map to the closest (lon, lat) in the dynamic_grid object for earch attack!")
                        #TODO, because it is bad if we put datapoints on land...
                        #TODO, we would also like to have all possible locations initialized also, especially with sailability and neighbours
                        raise NotImplementedError
                    
                    if (lon, lat) in states_info:
                        states_info[(lon,lat)]["n_attacks"]+=1
                        states_info[(lon,lat)]["danger"]+=sample_danger
                    else:
                        states_info[(lon,lat)]["n_attacks"]=1
                        states_info[(lon,lat)]["danger"]=sample_danger
    
    #for (lon,lat) in states_info.keys(): #adding the relative danger, maybe not needed as this is for an MDP
        #states_info[(lon,lat)]["relative_danger"]=states_info[(lon,lat)]["relative_danger"]/total_danger
    
    dataset={"min_lon":min_lon, "max_lon":max_lon, "min_lat":min_lat, "max_lat":max_lon, "total_danger":total_danger, "number_of_attacks":number_of_attacks, "state_data":states_info}
    return dataset

    #reverses in order to be plotable in different colors, nice 
    lon_lat_by_n_atacks={} #formated {number_of_atacks:(lons:list, lats:list)}
    for (lon,lat) in density_of_atacks.keys():
        number_of_atacks=density_of_atacks[(lon,lat)]
        if number_of_atacks in lon_lat_by_n_atacks:
            lon_lat_by_n_atacks[number_of_atacks][0].append(lon)
            lon_lat_by_n_atacks[number_of_atacks][1].append(lat)
        else:
            lon_lat_by_n_atacks[number_of_atacks]=([lon],[lat])

    return lon_lat_by_n_atacks, max_lat, min_lat, max_lon, min_lon


def plot_dataset_on_map(dataset, attribute="danger", Range:int=5):
    "supply attribute either danger or n_attacks, range 5 means 5 intervals as well as regions with 0, regions according to (]"
    lat_lons=dataset['state_data'].keys()
    if attribute=="danger":
        risk_to_plot=1
    #TODO dynamic ranges, give number of ranges for example and dividing it unformly
    # my_dict.values()

    #reverses in order to be easily plotable in different colors
    #lon_lat_by_n_atacks={} #formated {number_of_atacks:(lons:list, lats:list)}
    #for (lon,lat) in states_info.keys():
        #number_of_atacks=density_of_atacks[(lon,lat)]
        #if number_of_atacks in lon_lat_by_n_atacks:
            #lon_lat_by_n_atacks[number_of_atacks][0].append(lon)
            #lon_lat_by_n_atacks[number_of_atacks][1].append(lat)
        #else:
            #lon_lat_by_n_atacks[number_of_atacks]=([lon],[lat])

    #return lon_lat_by_n_atacks, max_lat, min_lat, max_lon, min_lon

    margin = 2 # buffer to add to the range
    lat_min = dataset['lat_min'] - margin
    lat_max = dataset['lat_max'] + margin
    lon_min = dataset['lon_min'] - margin
    lon_max = dataset['lon_max'] + margin

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



    if range:
        range_lons_lats={} #so can do the same thing but with things in a range...

    for i in lon_lat_by_n_atacks.keys():
        lons, lats=lon_lat_by_n_atacks[i]
        lons=np.array(lons) #here we array things
        lats=np.array(lats)
        if Range:
            for l in range(len(Range)-1):
                if i in range(Range[l], Range[l+1]):
                    if (Range[l],Range[l+1]) in range_lons_lats.keys():
                        range_lons_lats[(Range[l],Range[l+1])][0]=np.concatenate((range_lons_lats[(Range[l],Range[l+1])][0], lons))
                        range_lons_lats[(Range[l],Range[l+1])][1]=np.concatenate((range_lons_lats[(Range[l],Range[l+1])][1], lats))
                    else:
                        range_lons_lats[(Range[l],Range[l+1])]=[lons, lats]
        else:
            lons, lats = m(np.array(lons), np.array(lats)) #convert to map cordinates
            m.scatter(lons, lats, marker = 'o', s=5, zorder=5, label=f'{i} atacks') #zorder makes the dots on top of the map
    if Range:
        for (i,j) in range_lons_lats.keys():
            lons, lats=range_lons_lats[(i,j)]
            lons, lats = m(np.array(lons), np.array(lats)) #convert to map cordinates
            m.scatter(lons, lats, marker = 'o', s=5, zorder=5, label=f'{i}=<n atacks<{j}') #zorder makes the dots on top of the map

    plt.legend()
    plt.show()

if __name__=='__main__':
    dataset=read_data(approximate_to=2)
    plot_dataset_on_map(dataset)


