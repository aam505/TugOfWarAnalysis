import numpy as np
import matplotlib.pyplot as plt

data = np.random.rayleigh(scale=1, size=(30,4))
labels = list("ABCD")
colors = ["crimson", "purple", "limegreen", "gold"]

width=0.4
fig, ax = plt.subplots()
for i, l in enumerate(labels):
    x = np.ones(data.shape[0])*i + (np.random.rand(data.shape[0])*width-width/2.)
    ax.scatter(x, data[:,i], color=colors[i], s=25)
    mean = data[:,i].mean()
    ax.plot([i-width/2., i+width/2.],[mean,mean], color="k")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)

plt.show()