from mpl_toolkits.basemap import Basemap
import pandas as pd
import math

riskiest = -100
def riskCalc(location: tuple[float, float], riskFunc: callable = None) -> float:
    if riskFunc == None:
        return 1
    else:
        return riskFunc()
    
def generateFrame(lat: tuple[float, float] = (-90, 90), lon: tuple[float, float] = (-180, 180), scale: float = .5, riskFunc: callable = None) -> pd.DataFrame:
    map = Basemap()
    if lat[0] < -90 or lat [1] > 90:
        print("Lattitude out of bounds")
    if lon[0] < -180 or lon[1] > 180:
        print("Longitude out of bounds")
    lats = []
    lons = []
    for t in range(math.floor(lat[0] / scale), math.ceil(lat[1] / scale)):
        lats.append(t * scale)
    for n in range(math.floor(lon[0] / scale), math.ceil(lon[1] / scale)):
        lons.append(n * scale)
    df = pd.DataFrame(index=lons, columns=lats)
    for t in range(math.floor(lat[0] / scale), math.ceil(lat[1] / scale)):
        for n in range(math.floor(lon[0] / scale), math.ceil(lon[1] / scale)):
            lattitude = t * scale
            longitude = n * scale
            if not map.is_land(longitude, lattitude):
                df.loc[longitude, lattitude] = riskCalc((longitude, lattitude), None)
            else:
                df.loc[longitude, lattitude] = "-"
    return df

def loadFrame(filePath: str) -> pd.DataFrame:
    return pd.read_csv(filePath)



def main():
    lattitude = (-12.5, 19.5)
    longitude = (91, 155)
    scale = .5
    generateFrame(lattitude, longitude, scale).to_csv(f"{lattitude}_{longitude}_{scale}.csv")

if __name__ == "__main__":
    main()