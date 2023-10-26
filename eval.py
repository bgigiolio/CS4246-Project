import pandas as pd
from datetime import datetime, timedelta
import random
import math
import numpy

def expo_distribution(range: int):
    x = random.expovariate(1)
    inverted_x = 1 / (x + 1)
    result = math.ceil(inverted_x * range)
    return result

class Evaluator():

    def __init__(self, scale: float, epochs: int = 1, epoch_duration: int = 30, cargo_cost: int = 200000000, cost_per_move: int = 2000) -> None:
        #111 km per longitude
        self.dates = []
        self.scale = scale
        self.cargo_cost = cargo_cost
        self.cost_per_move = cost_per_move
        df = pd.read_csv("pirate_attacks.csv")
        for _ in range(epochs):
            fDay = datetime.strptime(df.iloc[0]["date"], '%Y-%m-%d')
            lDay = datetime.strptime(df.iloc[-1]["date"], '%Y-%m-%d')
            r = (lDay - fDay).days - epoch_duration
            delta = timedelta(days=expo_distribution(r))
            self.dates.append((fDay + delta, fDay + delta + timedelta(days=epoch_duration)))
        self.attacks = []
        for d in self.dates:
            mask = (df['date'] > str(d[0])) & (df['date'] <= str(d[1]))
            a = df.loc[mask]
            for index, row in a.iterrows():
                self.attacks.append((row['longitude'], row['latitude'], row['attack_type']))

    def generatePenalty(self):
        max_penalty = -100
        gradient_radius = 5 #in lat/lon steps
        penalties = {}
        a = numpy.array(range(0, int(5 / self.scale) + 1)) * self.scale * (max_penalty / 5)
        penaltyArr = a[::-1]
        for attack in self.attacks: #TODO: ask team about how to deal with scale .5 (do we round to .5 or use .5 as distance between points)
            miniPenalties = {}
            attackLonLat = (self.scale * round(attack[0]/self.scale), self.scale * round(attack[1]/self.scale))
            for x in range(0, int(gradient_radius / self.scale) + 1):
                for y in range(0, int(gradient_radius / self.scale) + 1):
                    if (x + y) <= gradient_radius / self.scale:
                        lon = round(float(attackLonLat[0]) + (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) + (y * self.scale), 4)
                        if (lon, lat) not in penalties:
                            penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            penalties[(lon, lat)] += penaltyArr[x + y]
                        lon = round(float(attackLonLat[0]) - (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) + (y * self.scale), 4)
                        if (lon, lat) not in penalties:
                            penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            penalties[(lon, lat)] += penaltyArr[x + y]
                        lon = round(float(attackLonLat[0]) + (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) - (y * self.scale), 4)
                        if (lon, lat) not in penalties:
                            penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            penalties[(lon, lat)] += penaltyArr[x + y]
                        lon = round(float(attackLonLat[0]) - (x * self.scale), 4)
                        lat = round(float(attackLonLat[1]) - (y * self.scale), 4)
                        if (lon, lat) not in penalties:
                            penalties[(lon, lat)] = penaltyArr[x + y]
                            miniPenalties[(lon, lat)] = penaltyArr[x + y]
                        elif (lon, lat) not in miniPenalties:
                            penalties[(lon, lat)] += penaltyArr[x + y]
        self.penalties = penalties




    def evalPath(self, path: tuple, index_to_coord: dict):
        return None
        

def main():
    Evaluator(.5, epochs=2).generatePenalty()
    

if __name__ == "__main__":
    main()