import csv
import numpy as np
import json
import pdb

with open("cfg/Box Measurements - Sheet1.csv", newline="") as csvfile:
    data = list(csv.reader(csvfile))
out_dict = []

for d in data[1:]:
    current_dict = {}
    current_dict["id"] = (int)(d[0])
    current_dict["dim"] = [float(d[1]) / 100, float(d[2]) / 100, float(d[3]) / 100]
    current_dict["alias"] = d[4]
    current_dict["color"] = list(np.random.random(3).round(3))
    out_dict.append(current_dict)

print(out_dict)
with open("cfg/boxes.json", "w") as w:
    json.dump(out_dict, w)
