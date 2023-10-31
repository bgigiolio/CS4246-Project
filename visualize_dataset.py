#from dataset import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import tqdm

#TODO: generate range so that equal amount of samples in each range

def plot_dataset_on_map(dataset, Attribute:str="danger", Ranges:int=5):
    "supply a state attributes which is a number such as danger or n_attacks, range 5 means 5 intervals as well as states with 0, regions according to (a,b].\
        Marks start, goal as >, < if defined."
    
    attribute_state_dict={}
    for state in tqdm.tqdm(dataset.states.keys(), desc="shuffling data"):
        if dataset.states[state][Attribute] in attribute_state_dict:
            attribute_state_dict[dataset.states[state][Attribute]].append(state)
        else:
            attribute_state_dict[dataset.states[state][Attribute]]=[state]

    #creating the range cutoffs:
    attributes=list(attribute_state_dict.keys())
    
    range_cuts=[i*max(attributes)/Ranges for i in range(Ranges+1)]

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
    
    if dataset.goal:
        goal_lon, goal_lat=dataset.goal
        start_lon, start_lat=dataset.start
        goal_lon, goal_lat=m(goal_lon, goal_lat)
        start_lon, start_lat=m(start_lon, start_lat)
        m.scatter(start_lon, start_lat, marker = '>', s=7, zorder=5, label="start")
        m.scatter(goal_lon, goal_lat, marker = '>', s=7, zorder=5, label="goal")
    
    plt.legend()
    plt.show()




