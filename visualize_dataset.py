import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import tqdm

# if these types of figures will actually be used in the final report: 
#TODO: generate range so that equal amount of samples in each range
#TODO: crop the pictures to a certain size with no blank space. Save only the legend with purely white background...

def plot_dataset_on_map(dataset, Attribute:str="danger", Ranges:int=5, size=2, legend_size=10, Legend=True):
    "supply a state attributes which is a number such as, density danger or n_attacks, range 5 means 5 intervals as well as states with 0, regions according to (a,b].\
        Marks start, goal as >, < if defined. Use no legend if want to show the legend in the report in another way"
    
    plt.rc('legend',fontsize=legend_size) # using a size in points

    attribute_state_dict={}
    for state in tqdm.tqdm(dataset.states.keys(), desc="shuffling data"):
        if dataset.states[state][Attribute] in attribute_state_dict:
            attribute_state_dict[float(dataset.states[state][Attribute])].append(state)
        else:
            attribute_state_dict[float(dataset.states[state][Attribute])]=[state]

    #creating the range cutoffs:
    attributes=list(attribute_state_dict.keys())
    if Attribute=="action":
        range_cuts=[0,1,2,3]
    else:
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
    
    if float(0) in attribute_state_dict.keys(): #if there is states with 0 seperate those states
        lons_0=[]
        lats_0=[]
        for j in range(len(attribute_state_dict[0])):
            lons_0.append(attribute_state_dict[0][j][0])
            lats_0.append(attribute_state_dict[0][j][1])
        lons_lats_by_range={(0,0):[lons_0, lats_0]} #format (a,b]
    else:
        lons_lats_by_range={} #format (a,b]

    for i in range(len(range_cuts)-1):
        #initialising every range
        lons_lats_by_range[(range_cuts[i], range_cuts[i+1])]=[[], []]

    for attribute in attribute_state_dict:
        for (a,b) in lons_lats_by_range.keys():
            if a<attribute<=b: #if the attribute in this range split all the states with that attribute into lons and lats lists and add to the range dict
                lons_lats_by_range[(a, b)][0]+=[attribute_state_dict[attribute][j][0] for j in range(len(attribute_state_dict[attribute]))]
                lons_lats_by_range[(a, b)][1]+=[attribute_state_dict[attribute][j][1] for j in range(len(attribute_state_dict[attribute]))]
    Color=['lime', 'teal', 'r', 'y', 'pink', 'c', 'm', 'k', ] #to make sure no blue on blue
    for (i,(a,b)) in enumerate(lons_lats_by_range.keys()):
        [lons, lats]=lons_lats_by_range[(a,b)]
        lons, lats = m(np.array(lons), np.array(lats)) #convert to map cordinates! done!
        if Attribute=="action":
            mapping={0:"right", 1:"up", 2:"left", 3:"down"}
            m.scatter(lons, lats, marker = 'o', s=size, zorder=5, label=f'{Attribute} is {mapping[b]}', color=Color[i]) 
        elif a==b:
            m.scatter(lons, lats, marker = 'o', s=size, zorder=5, label=f'{Attribute} is {a}', color=Color[i]) #zorder makes the dots on top of the map
        else:
            m.scatter(lons, lats, marker = 'o', s=size, zorder=5, label=f'{a}<{Attribute}<={b}', color=Color[i]) #zorder makes the dots on top of the map
        
    if dataset.goal:
        goal_lon, goal_lat=dataset.goal
        start_lon, start_lat=dataset.start
        goal_lon, goal_lat=m(goal_lon, goal_lat)
        start_lon, start_lat=m(start_lon, start_lat)
        m.scatter(start_lon, start_lat, marker = '>', s=size+3, zorder=5, label="start", color=Color[-1])
        m.scatter(goal_lon, goal_lat, marker = '>', s=size+3, zorder=5, label="goal", color=Color[-2])
    if Legend:
        plt.legend()
    plt.show()




