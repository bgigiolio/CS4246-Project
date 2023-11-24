import pandas as pd
from datetime import datetime, timedelta
import random
import math
import numpy
import matplotlib.pyplot as plt
import tqdm
import rasterio
from dataset import get_index, local_average

def expo_distribution(range: int):
    x = random.expovariate(1)
    inverted_x = 1 / (x + 1)
    result = math.ceil(inverted_x * range)
    return result

class Evaluator():
    """
    A class that allows policy evaluation

    Attributes:
    ___________
    scale: float
        This represents the scale of the MDP with degrees of latitude as the unit
    dates: list[dateTime]
        This is the list of dates of attacks sampled
    cargo_cost: int
        The cost of the container ship if lost
    cost_per_move: int
        The added dollar cost of each step taken. Meant to represent fuel costs.
    epoch_duration: int
        The amount of days of pirate attacks to be sampled
    epochs: int
        Amount of epoch_durations to sample
    df: Dataframe
        Dataframe of total pirate attacks
    
    Functions:
    ___________
    generateAttacks() -> None
        Generates a list of attacks from [epochs] samples of [epoch_duration] time
    generatePenalty() -> None
        Generates a dict of all lattitudes that contain a piracy penalty, and the penalty associated with it
    regenerate() -> None
        Runs generateAttacks() and genereatePenalty()
        Can be used to create a new sample to evaluate on without constructing an entirely new Evaluator
    evalPolicy(path: tuple, index_to_coord: dict) -> int
        Takes in a policy and its corresponding index to coord list to evaluate performance on the current sampled piracy attacks.
    """
    def __init__(self, scale: float, epochs: int = 1, epoch_duration: int = 30, cargo_cost: int = 200000000, cost_per_move: int = 16568) -> None:
        #111 km per longitude
        self.dates = []
        self.scale = scale
        self.cargo_cost = cargo_cost
        self.cost_per_move = cost_per_move
        self.epochs = epochs
        self.epoch_duration = epoch_duration
        self.df = pd.read_csv("pirate_attacks.csv")
        self.regenerate()

    # def rawScore(self)

    def generateAttacks(self):
        self.dates = []
        for _ in range(self.epochs):
            fDay = datetime.strptime(self.df.iloc[0]["date"], '%Y-%m-%d')
            lDay = datetime.strptime(self.df.iloc[-1]["date"], '%Y-%m-%d')
            r = (lDay - fDay).days - self.epoch_duration
            delta = timedelta(days=expo_distribution(r))
            self.dates.append((fDay + delta, fDay + delta + timedelta(days=self.epoch_duration)))
        self.attacks = []
        for d in self.dates:
            mask = (self.df['date'] > str(d[0])) & (self.df['date'] <= str(d[1]))
            a = self.df.loc[mask]
            for _, row in a.iterrows():
                self.attacks.append((row['longitude'], row['latitude'], row['attack_type']))
    
    def generateAllAttacks(self):
        self.dates = []
        fDay = datetime.strptime(self.df.iloc[0]["date"], '%Y-%m-%d')
        lDay = datetime.strptime(self.df.iloc[-1]["date"], '%Y-%m-%d')
        r = (lDay - fDay).days
        # delta = timedelta(days=expo_distribution(r))
        self.dates.append((fDay, lDay))
        self.attacks = []
        for d in self.dates:
            mask = (self.df['date'] > str(d[0])) & (self.df['date'] <= str(d[1]))
            a = self.df.loc[mask]
            for _, row in a.iterrows():
                self.attacks.append((row['longitude'], row['latitude'], row['attack_type']))

    def generatePenalty(self):
        max_penalty = -100
        gradient_radius = 5 #in lat/lon steps
        self.penalties = {}
        a = numpy.array(range(0, int(5 / self.scale) + 1)) * self.scale * (max_penalty / 5)
        penaltyArr = a[::-1]
        for attack in tqdm.tqdm(self.attacks, desc="Generate penalty"): 
            miniPenalties = {}
            attackLonLat = (self.scale * round(attack[0]/self.scale), self.scale * round(attack[1]/self.scale))
            for x in range(0, int(gradient_radius / self.scale) + 1):
                for y in range(0, int(gradient_radius / self.scale) + 1):
                    if (x + y) <= gradient_radius / self.scale:
                        lon = round(float(attackLonLat[0]) + (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) + (y * self.scale), 4)
                        if (lon, lat) not in self.penalties:
                            self.penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            self.penalties[(lon, lat)] += penaltyArr[x + y]
                        lon = round(float(attackLonLat[0]) - (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) + (y * self.scale), 4)
                        if (lon, lat) not in self.penalties:
                            self.penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            self.penalties[(lon, lat)] += penaltyArr[x + y]
                        lon = round(float(attackLonLat[0]) + (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) - (y * self.scale), 4)
                        if (lon, lat) not in self.penalties:
                            self.penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            self.penalties[(lon, lat)] += penaltyArr[x + y]
                        lon = round(float(attackLonLat[0]) - (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) - (y * self.scale), 4)
                        if (lon, lat) not in self.penalties:
                            self.penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            self.penalties[(lon, lat)] += penaltyArr[x + y]
    def regenerate(self):
        self.generateAllAttacks()
        self.generatePenalty()

    def evalPolicy(self, pathDict: dict, log: bool = False) -> int:
        score = 0
        file_path = 'ShipDensity_Commercial1.tif'
        raster_data = rasterio.open(file_path) 
        print("reading the huge shipping density file...")   
        raster_array = raster_data.read(1)
        path = pathDict["path"]
        for i in range(len(path)):
            lon,lat = path[i]
            colon_index=get_index(lon, lon_input=True)
            row_index=get_index(lat, lon_input=False)
            density=local_average(self, row_index,colon_index, raster_array, int(self.scale/(0.005)), "local_averege")
            if density > 1:
                if path[i] in self.penalties.keys():
                    score += self.penalties[path[i]] / density
            elif path[i] in self.penalties.keys():
                score += self.penalties[path[i]] *.001
        if log:
            f = open(f"Evaluations\\{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.txt", "w")
            f.write("_____Path Evaluation_____\n")
            f.write(f"Params: Cargo Cost: {self.cargo_cost} | Cost per Move: {self.cost_per_move}\n")
            f.write(f"Distance: {pathDict['distance']} | Distance Cost: {pathDict['distance'] * self.cost_per_move * self.scale}\n")
            f.write(f"Piracy Penalty: {score} | Money Adjusted Score: {score * (self.cargo_cost / 100000)}\n")
            f.write(f"Total penalty: ${score * (self.cargo_cost / 100000) - pathDict['distance'] * self.cost_per_move} \n")
        return score * (self.cargo_cost / 100000) - pathDict['distance'] * self.cost_per_move

        return score + pathDict["distance"]
    
    def largeAverage(self, iterations, pathDict: dict):
        total = 0
        totalOverTime = []
        for i in tqdm.tqdm(range(iterations), desc="Evaluating Path"):
            self.regenerate()
            total += self.evalPolicy(pathDict)
            totalOverTime.append(total / (i + 1))
            x = range(1, iterations + 1)
        f = plt.figure()
        f.set_figheight(10)
        f.set_figwidth(6)
        plt.plot(x, totalOverTime)
        plt.title(f'Average Penalty over {iterations} Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Average Penalty')
        plt.autoscale()
        # f.autoscale()
        plt.show()
        

def main():
    Evaluator(.5, epochs=2)
    

if __name__ == "__main__":
    main()