from mpl_toolkits.basemap import Basemap
import pandas as pd
import math
import os
from dataset import Dataset
import tqdm
# TODO: USE HIGHER REZ MAP
def riskCalcNone(lon: float, lat: float, data: Dataset = None) -> float:
    return 1

def riskCalcNeighbors(lon: float, lat: float, data: Dataset):
    if (round(lon, 4), round(lat, 4)) in data.states:
        return data.states[(round(lon, 4), round(lat, 4))]["danger"]
    else:
        # print((round(lon, 4), round(lat, 4))
        return "-"
    
def generateFrame(lat: tuple[float, float] = (-90, 90), lon: tuple[float, float] = (-180, 180), scale: float = .5, riskFunc: callable = riskCalcNone, data: Dataset = None) -> pd.DataFrame:
    """
    Generates a dataframe of size |lat[1] - lat[0| x |lon[1] - lon[0]| * 1/scale
    Columns correspond to lattitudes
    Rows/indices correspond to longitudes
    The values of each position is decided by riskFunc(lon, lat)
    """
    # if os.path.isfile(f"riskMaps/{lat}_{lon}_{scale}.csv"):
    #     return loadFrame(f"riskMaps/{lat}_{lon}_{scale}.csv")
    map = Basemap(resolution="h")
    if lat[0] < -90 or lat [1] > 90:
        print("Lattitude out of bounds")
    if lon[0] < -180 or lon[1] > 180:
        print("Longitude out of bounds")
    lats = []
    lons = []
    for t in range(math.ceil(lat[0] / scale), math.floor(lat[1] / scale)):
        lats.append(t * scale)
    for n in range(math.ceil(lon[0] / scale), math.floor(lon[1] / scale)):
        lons.append(n * scale)
    df = pd.DataFrame(index=lons, columns=lats)
    for t in tqdm.tqdm(range(math.ceil(lat[0] / scale), math.floor(lat[1] / scale)), desc="test"):
        for n in range(math.ceil(lon[0] / scale), math.floor(lon[1] / scale)):
            lattitude = t * scale
            longitude = n * scale
            if not map.is_land(longitude, lattitude):
                df.loc[longitude, lattitude] = riskFunc(longitude, lattitude, data)
            else:
                df.loc[longitude, lattitude] = "-"
    return df

def loadFrame(filePath: str) -> pd.DataFrame:
    """
    Returns a dataframe stored at filePath
    """
    return pd.read_csv(filePath)

def updateFrame(df: pd.DataFrame, lats: list[float], lons: list[float], vals: [float]) -> int:
    """
    Updates dataframe df at positions (lons[i], lats[i]) with the value at vals[i]

    Requirements: len(lats) = len(lons) = len(vals)
        -Each index of lattitude matches with the same index of longitude and will be set to 
         corresponding value in vals
    """
    if not (len(lats) == len(lons) and len(lons) == len(vals)):
        return -1
    try:
        for i in range(len(lats)):
            df.loc[lons[i], lats[i]] = vals[i]
    except:
        return -1
    return 1

def updateFrameByFunc(df: pd.DataFrame, lats: list[float], lons: list[float], func: callable) -> int:
    """
    Updates dataframe df at positions (lons[i], lats[i]) using a function that takes lattitude and 
    longitude as input

    Requirements: len(lats) = len(lons)
        -Each index of lattitude matches with the same index of longitude
    """
    if not len(lats) == len(lons):
        return -1
    try:
        for i in range(len(lats)):
            df.loc[lons[i], lats[i]] = func(lats[i], lons[i], df.loc[lons[i], lats[i]])
    except:
        return -1
    return 1


def main():
    lattitude = (-12.5, 31.5)
    longitude = (88.5, 153)
    scale = .5
    dataset=Dataset(88.5, 153, -12.5, 31.5) #South East Asia
    dataset.generate_states(distance=scale) #needs to be done first
    dataset.load_pirate_data(spread_of_danger=1)
    dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
    generateFrame(lattitude, longitude, scale, riskCalcNeighbors, dataset).to_csv(f"riskMaps/{lattitude}_{longitude}_{scale}.csv")

if __name__ == "__main__":
    main()