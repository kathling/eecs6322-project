import numpy as np

# normalize x to range [-1, 1]
def normalize(x):
    min = np.min(x)
    max = np.max(x)
    return 2*(x-min)/(max-min) - 1

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
print(normalize(x))