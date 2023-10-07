import csv
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

#stolen from https://stackoverflow.com/questions/44488167/plotting-lat-long-points-using-basemap

south_east_asia={"min_lat":-11.0, "max_lat":25.0, "min_lon":90.0, "max_lon":145.0}

def read_dataset(region=south_east_asia, path='pirate_attacks.csv', approximate_to_int=False):
    "by deafult region south_east_asia={min_lat:-11.0, max_lat:25.0, min_lon:90.0, max_lon:145.0}, path pirate_attacks.csv, no approximations"
    #lats=[]
    #lons=[]
    min_lon=region['min_lon']
    max_lon=region['max_lon']
    min_lat=region['min_lat']
    max_lat=region['max_lat']

    density_of_atacks={} #formated (lat, lon):number of attacks, most usefull if approximate to int (or some other filtering)
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
                if approximate_to_int:
                    lon=round(lon, 0)
                    lat=round(lat, 0)
                if min_lon<lon<max_lon and min_lat<lat<max_lat:
                    if (lon, lat) in density_of_atacks:
                        density_of_atacks[(lon,lat)]+=1
                    else:
                        density_of_atacks[(lon,lat)]=1
                    #lons.append(lon)
                    #lats.append(lat)
    
    #reverses in order to be plotable in different colors
    lon_lat_by_n_atacks={} #formated {number_of_atacks:(lons:list, lats:list)}
    for (lon,lat) in density_of_atacks.keys():
        number_of_atacks=density_of_atacks[(lon,lat)]
        if number_of_atacks in lon_lat_by_n_atacks:
            lon_lat_by_n_atacks[number_of_atacks][0].append(lon)
            lon_lat_by_n_atacks[number_of_atacks][1].append(lat)
        else:
            lon_lat_by_n_atacks[number_of_atacks]=([lon],[lat])

    return lon_lat_by_n_atacks, max_lat, min_lat, max_lon, min_lon


def plot_piracy_attacks_on_map(lon_lat_by_n_atacks:dict, lat_max:float, lat_min:float, lon_max:float, lon_min:float, Range:list=False):
    "supply the lat_max, lat_min, lon_max of the dataset, range in the format [1, i:int, j:int, k:int], plotting i=< < j etc"

    #TODO calculate lat_max, lat_min etc from the dict
    # my_dict.values()

    # determine range to print based on min, max lat and lon of the data, only the region we are interested in
    margin = 2 # buffer to add to the range
    lat_min = lat_min - margin
    lat_max = lat_max + margin
    lon_min = lon_min - margin
    lon_max = lon_max + margin

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
    #m.drawstates() #American states...
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color = 'white',lake_color='#46bcec') #color of oceans etc
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
    lon_lat_by_n_atacks, max_lat, min_lat, max_lon, min_lon=read_dataset(approximate_to_int=True)
    plot_piracy_attacks_on_map(lon_lat_by_n_atacks, max_lat, min_lat, max_lon, min_lon, Range=[1, 10, 20, 200])


