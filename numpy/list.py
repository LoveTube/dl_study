import csv
import numpy as np


with open("winequality-red.csv", 'r') as f:
	wines =  list(csv.reader(f, delimiter=";"))
wines = np.array(wines[1:], dtype=np.float)

print(wines)
print(wines.shape)
