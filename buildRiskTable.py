from mpl_toolkits.basemap import Basemap
import pandas as pd
import math
import os
from dataset import Dataset
import tqdm
import json

def riskCalcNone(lon: float, lat: float, data: Dataset = None) -> float:
    return 1

def riskCalcNeighbors(lon: float, lat: float, data: Dataset):
    if (round(lon, 4), round(lat, 4)) in data.states:
        return data.states[(round(lon, 4), round(lat, 4))]["danger"]
    else:
        # print((round(lon, 4), round(lat, 4))
        return "-"
    
def riskCalcNeighborsGoal(lon: float, lat: float, data: Dataset, goal: tuple[float, float]):
    if (round(lon, 4), round(lat, 4)) in data.states:
        distance = math.sqrt(abs(goal[0] - lon)**2 + abs(goal[1] - lat)**2)
        return data.states[(round(lon, 4), round(lat, 4))]["danger"] - distance
    else:
        # print((round(lon, 4), round(lat, 4))
        return "-"
    
#TODO: restructure to be MDP
#TODO: create way to load from JSON
#TODO: set up goal and start
class MDP:
    """
    A class to represent an MDP instance
    
    Attributes:
    ____________
    indexToCoord: dict
        A dictionary of states where the key is the state's number/index and the value is the coordinate
        eg: {0: (45, 45)}
    coordToIndex: dict
        A dictionary of longitudes, lattitudes, and state numbers/indices
        Structure: {longitude: {lattitude: index}}
        eg: {45: {45: 0}}
        to access: index = MDP.coordToIndex[longitude][lattitude]
    lon: tuple
        A tuple of the longitude range covered by the MDP
        eg: (-180, 180)
    lat: tuple
        A tuple of the lattitude range covered by the MDP
        eg: (-90, 90)
    scale: float
        The percision used by the MDP
    filepath: str
        The local filepath that stores the riskMap and JSON representation of the class if the class has been archived


    """
    def __init__(self, lon: tuple[float, float] = (-180, 180), lat: tuple[float, float] = (-90, 90),  scale: float = .5, riskFunc: callable = riskCalcNeighbors, data: Dataset = None, JSON_file: dict = None) -> pd.DataFrame:
        """
        Generates 
        """
        # TODO: store json and CSV
        if JSON_file:
            f = open({JSON_file},) 
            JSON = json.load(f)
            if JSON.has_key("lat") and JSON.has_key("indexToCoord"):
                self.indexToCoord = JSON["indexToCoord"]
                self.coordToIndex = JSON["coordToIndex"]
                self.lat = JSON["lat"]
                self.lon = JSON["lon"]
                self.scale = JSON["scale"]
                self.filepath = JSON["filepath"]
                # df = JSON["df"]
                return
        if os.path.isfile(f"riskMaps/{lat}_{lon}_{scale}/risk.csv"):
            df = self.loadFrame(f"riskMaps/{lat}_{lon}_{scale}")
            self.filepath = f"riskMaps/{lat}_{lon}_{scale}"
        else:
            # map = Basemap(resolution="h")
            map = Basemap()

            if lat[0] < -90 or lat [1] > 90:
                raise Exception("Lattitude out of range")
            if lon[0] < -180 or lon[1] > 180:
                raise Exception("Longitude out of range")
            lats = []
            lons = []
            for t in range(math.ceil(lat[0] / scale), math.floor(lat[1] / scale)):
                # lats.append(t * scale)
                lats.append(round((t * scale), 4))
            for n in range(math.ceil(lon[0] / scale), math.floor(lon[1] / scale)):
                # lons.append(n * scale)
                lons.append(round((n * scale), 4))
            df = pd.DataFrame(index=lons, columns=lats)
            self.indexToCoord = {}
            self.coordToIndex = {}
            counter = 0
            for n in tqdm.tqdm(range(math.ceil(lon[0] / scale), math.floor(lon[1] / scale)), desc="Generating Frame"):
                longitude = round(n * scale, 4)
                self.coordToIndex[longitude] = {}
                for t in range(math.ceil(lat[0] / scale), math.floor(lat[1] / scale)):
                    lattitude = round(t * scale, 4)
                    self.indexToCoord[counter] = (longitude, lattitude)
                    self.coordToIndex[longitude][lattitude] = counter
                    counter += 1
                    if not map.is_land(longitude, lattitude):
                        df.loc[longitude, lattitude] = riskFunc(longitude, lattitude, data)
                        # df.loc[longitude, lattitude] = 1
                    else:
                        df.loc[longitude, lattitude] = "-"
            self.lat = lat
            self.lon = lon
            self.scale = scale
            try:  
                os.mkdir(f"riskMaps/{self.lat}_{self.lon}_{self.scale}")  
            except OSError as error: 
                print("Folder already exists, skipping...")  
            self.filepath = f"riskMaps/{self.lat}_{self.lon}_{self.scale}"
            self.generateCSV(df=df)

    def toJSON(self):
        j = json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
        f = open(f"{self.filepath}/JSON.json", "w")
        f.write(j)

    def generateCSV(self, df: pd.DataFrame):
        df.to_csv(f"{self.filepath}/risk.csv")

    def loadFrame(self, filePath: str) -> pd.DataFrame:
        """
        Returns a dataframe stored at filePath
        """
        self.filepath = filePath
        return pd.read_csv(f"{filePath}/risk.csv")

    def updateFrame(self, df, lats: list[float], lons: list[float], vals: [float]) -> int:
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

    def updateFrameByFunc(self, df, lats: list[float], lons: list[float], func: callable) -> int:
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
                df.loc[lons[i], lats[i]] = func(lats[i], lons[i], self.df.loc[lons[i], lats[i]])
        except:
            return -1
        return 1
    def archive(self):
        self.toJSON()
        self = None


def main():
    lattitude = (-12.5, 20)
    longitude = (88.5, 100)
    scale = .5
    dataset=Dataset(longitude[0], longitude[1], lattitude[0], lattitude[1]) #South East Asia
    dataset.generate_states(distance=scale) #needs to be done first
    dataset.load_pirate_data(spread_of_danger=1)
    dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
    a = MDP(lat=lattitude, lon=longitude, scale=scale, data=dataset)
    a.archive()

if __name__ == "__main__":
    main()