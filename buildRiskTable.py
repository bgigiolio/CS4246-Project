from mpl_toolkits.basemap import Basemap
import pandas as pd
import math
import os
from dataset import Dataset
import tqdm
import json

def riskCalcNone(lon: float, lat: float, data: Dataset = None) -> float:
    """
    All lon, lat pairs mapped to 1
    """
    return 1

def riskCalcNeighbors(lon: float, lat: float, data: Dataset):
    """
    Generates riskMap using dataset's 'danger' value
    """
    if (round(lon, 4), round(lat, 4)) in data.states:
        return data.states[(round(lon, 4), round(lat, 4))]["danger"]
    else:
        return "-"
    
def riskCalcNeighborsGoal(lon: float, lat: float, data: Dataset, goal: tuple[float, float] = None):
    """
    Generates riskMap using dataset's 'danger' value and distance to goal
    """
    if goal is None:
        return riskCalcNeighbors(lon, lat, data)
    goalReward = 100
    if goal[0] == lon and goal[1] == lat:
        return goalReward
    if (round(lon, 4), round(lat, 4)) in data.states:
        distance = math.sqrt(abs(goal[0] - lon)**2 + abs(goal[1] - lat)**2)
        return round(data.states[(round(lon, 4), round(lat, 4))]["danger"] - distance, 4)
    else:
        return "-"
    
def as_float(obj: dict):
    """Checks each dict passed to this function if it contains the key "value"
    Args:
        obj (dict): The object to decode

    Returns:
        dict: The new dictionary with changes if necessary
    """
    return {round(float(key), 4) : (as_float(value) if isinstance(value, dict) else value) for key, value in obj.items()}

def as_int(obj: dict):
    """Converts string keys to integers
    Args:
        obj (dict): The object to decode

    Returns:
        dict: The new dictionary with changes if necessary
    """
    return {int(key) : tuple(value) for key, value in obj.items()}


    
class MDP:
    """
    A class to represent an MDP instance
    
    Attributes:
    ____________
    indexToCoord: dict
        A dictionary of states where the key is the state's number/index and the value is the coordinate
        eg: {0: (45, 45)}
    coordToIndex: dict
        A dictionary of longitudes, latitudes, and state numbers/indices
        Structure: {longitude: {latitude: index}}
        eg: {45: {45: 0}}
        to access: index = MDP.coordToIndex[longitude][latitude]
    lon: tuple
        A tuple of the longitude range covered by the MDP
        eg: (-180, 180)
    lat: tuple
        A tuple of the latitude range covered by the MDP
        eg: (-90, 90)
    scale: float
        The percision used by the MDP
    filepath: str
        The local filepath that stores the riskMap and JSON representation of the class if the class has been archived
    """
    def __init__(self, lon: tuple[float, float] = (-180, 180), lat: tuple[float, float] = (-90, 90),  scale: float = .5, riskFunc: callable = riskCalcNeighborsGoal, data: Dataset = None, JSON_file: str = None, goal: tuple[float, float] = None) -> pd.DataFrame:
        """
        Generates a class to represent an MDP instance

        Parameters: 
        ___________
        lon: tuple(float, float)
            The desired longitude range of the MDP
            Defaults to (-180, 180)
        lat: tuple(float, float)
            The desired latitude range of the MDP
            Defaults to (-90, 90)
        scale: float
            The desired latitude/longitude percision of the MDP
            Defaults to .5
        riskFunc: callable
            The function for calculating risk. Must be a funciton with the following parameters:
            func(lon: float, lat: float, data: Dataset)
            or
            func(lon: float, lat: float, data: Dataset, goal: tuple[float, float] = None)
            Defaults to riskCalcNeighbors (riskCalcNeighborsGoal if goal param is filled)
        data: Dataset
            The data used by the risk function
            Defaults to None (Only do this if loading from JSON)
        JSON_file: str
            The filepath to a JSON file to load an MDP object from
            If this param is included, all other params will be ignored and the MDP will be generated from the file given
            Defaults to None (No file loaded, init carries on normally)
        goal: tuple(float, float)
            The goal location for the MDP
            Defaults to None (MDP will proceed with no goal)
        """
        if JSON_file or os.path.isfile(f"riskMaps/{lat}_{lon}_{scale}/JSON.json"):
            if not JSON_file:
                JSON_file = f"riskMaps/{lat}_{lon}_{scale}/JSON.json"
            f = open(JSON_file,) 
            JSON = json.load(f)
            self.indexToCoord = as_int(JSON["indexToCoord"])
            self.coordToIndex = as_float(JSON["coordToIndex"])
            self.lat = JSON["lat"]
            self.lon = JSON["lon"]
            self.scale = JSON["scale"]
            self.filepath = JSON["filepath"]
            self.goal = JSON["goal"]
            return
        else:
            map = Basemap()

            if lat[0] < -90 or lat [1] > 90:
                raise Exception("latitude out of range")
            if lon[0] < -180 or lon[1] > 180:
                raise Exception("Longitude out of range")
            lats = []
            lons = []
            for t in range(math.ceil(lat[0] / scale), math.floor(lat[1] / scale)):
                lats.append(round((t * scale), 4))
            for n in range(math.ceil(lon[0] / scale), math.floor(lon[1] / scale)):
                lons.append(round((n * scale), 4))
            df = pd.DataFrame(index=lons, columns=lats)
            self.indexToCoord = {}
            self.coordToIndex = {}
            counter = 0
            for n in tqdm.tqdm(range(math.ceil(lon[0] / scale), math.floor(lon[1] / scale)), desc="Generating Frame"):
                longitude = round(n * scale, 4)
                self.coordToIndex[longitude] = {}
                for t in range(math.ceil(lat[0] / scale), math.floor(lat[1] / scale)):
                    latitude = round(t * scale, 4)
                    self.indexToCoord[counter] = (longitude, latitude)
                    self.coordToIndex[longitude][latitude] = counter
                    counter += 1                       
                    if not map.is_land(longitude, latitude):
                        df.loc[longitude, latitude] = riskFunc(longitude, latitude, data, goal)
                    else:
                        df.loc[longitude, latitude] = "-"
            self.lat = lat
            self.lon = lon
            self.scale = scale
            self.goal = goal
            try:  
                os.mkdir(f"riskMaps/{self.lat}_{self.lon}_{self.scale}")  
            except OSError as error: 
                print(error)  
            self.filepath = f"riskMaps/{self.lat}_{self.lon}_{self.scale}"
            self.generateCSV(df=df)
            self.toJSON()

    def toJSON(self):
        """
        Dumps a JSON version of the class into the riskMaps folder
        """
        j = json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
        f = open(f"{self.filepath}/JSON.json", "w")
        f.write(j)

    def generateCSV(self, df: pd.DataFrame):
        """
        Generates a CSV of the riskMap to the riskMaps folder
        """
        df.to_csv(f"{self.filepath}/risk.csv")

    def loadFrame(self, filePath: str = None) -> pd.DataFrame:
        """
        Returns a dataframe stored at filePath
        """
        if filePath: 
            self.filepath = filePath
        else:
            filePath = self.filepath
        return pd.read_csv(f"{filePath}/risk.csv")

    def updateFrame(self, lats: list[float], lons: list[float], vals: [float], df: pd.DataFrame = None) -> int:
        """
        Updates dataframe df at positions (lons[i], lats[i]) with the value at vals[i]

        Requirements: len(lats) = len(lons) = len(vals)
            -Each index of latitude matches with the same index of longitude and will be set to 
            corresponding value in vals
        """
        if not df:
            df = self.loadFrame()
        if not (len(lats) == len(lons) and len(lons) == len(vals)):
            return -1
        try:
            for i in range(len(lats)):
                df.loc[lons[i], lats[i]] = vals[i]
        except:
            return -1
        return 1

    def updateFrameByFunc(self, lats: list[float], lons: list[float], func: callable, df: pd.DataFrame = None) -> int:
        """
        Updates dataframe df at positions (lons[i], lats[i]) using a function that takes latitude and 
        longitude as input

        Requirements: len(lats) = len(lons)
            -Each index of latitude matches with the same index of longitude
        """
        if not df:
            df = self.loadFrame()
        if not len(lats) == len(lons):
            return -1
        try:
            for i in range(len(lats)):
                df.loc[lons[i], lats[i]] = func(lats[i], lons[i], self.df.loc[lons[i], lats[i]])
        except:
            return -1
        return 1

def main():
    latitude = (-12.5, 31.5)
    longitude = (88.5, 152.9)
    scale = .5
    dataset=Dataset(longitude[0], longitude[1], latitude[0], latitude[1]) #South East Asia
    dataset.generate_states(distance=scale) #needs to be done first
    dataset.load_pirate_data(spread_of_danger=1)
    dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
    a = MDP(lat=latitude, lon=longitude, scale=scale, data=dataset, goal=(95, -5.5))
    a.toJSON()
    b = MDP(JSON_file="riskMaps\(-12.5, 20)_(88.5, 100)_0.5\JSON.json")
    utility = b.loadFrame()

if __name__ == "__main__":
    main()