import numpy as np

denominator = np.array([[0, 1], [0, 1]])
numerator = np.ones((2,2))

print(np.where(denominator != 0, numerator / denominator, 0))