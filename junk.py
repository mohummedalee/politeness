import numpy as np
import scipy.stats as stats
from collections import OrderedDict

data = np.genfromtxt('batch_titles_poll.csv', dtype=None, delimiter=',')
awards_num = data.shape[1]
awards = {}
for i in xrange(3, awards_num):
    awards[data[0, i]] = {}
    details = stats.itemfreq(data[1:, i])
    for name, count in details:
        awards[data[0, i]][name] = int(count)
    awards[data[0, i]] = OrderedDict(sorted(awards[data[0, i]].items(), key=lambda x: x[1], reverse=True))

with open("results_poll.txt", "w") as f:
    for title, details in awards.items():
        f.write(title + "\n")
        for name, count in details.items():
            f.write(name + "\t" + "\n")
        f.write("\n")
