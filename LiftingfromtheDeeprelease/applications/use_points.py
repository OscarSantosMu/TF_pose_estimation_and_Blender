from types import prepare_class
import numpy as np
import json

with open("points.json", 'r') as f:
    lst = json.load(f)

print(lst)
print(len(lst))

a_new = np.array(lst)

print(a_new)
print(a_new.shape)

#for i in range(a_new[0]):