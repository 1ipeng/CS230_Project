import numpy as np
DATA_BINS = "bins_529"

bin_dict = []
bin_size = 10
for a in np.arange(-110, 120, bin_size):
	for b in np.arange(-110, 120, bin_size):
		bin_dict.append([a, b])
np.save("bins_529", bin_dict)