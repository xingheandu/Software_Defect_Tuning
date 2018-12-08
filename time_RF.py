import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 14
sequential = (
    289.59, 329.72, 378.19, 497.01, 303.91, 325.11, 278.15, 294.12, 352.59, 3664.87, 2467.39, 328.04, 345.98, 483.84)
master_slave = (26.28, 28.73, 33.77, 44.01, 28.89, 32.23, 26.93, 26.24, 30.53, 447.05, 202.58, 30.75, 29.93, 53.01)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, sequential, bar_width,
                 alpha=opacity,
                 color='SkyBlue',
                 label='Sequential')

rects2 = plt.bar(index + bar_width, master_slave, bar_width,
                 alpha=opacity,
                 color='IndianRed',
                 label='Master-Slave')

plt.xlabel('Dataset')
plt.ylabel('Seconds')
plt.title('Tuning Time of Random Forest')
plt.xticks(rotation=90)
plt.xticks(index + bar_width, (
    'ant-1.3', 'ant-1.4', 'ant-1.6', 'camel-1.2', 'jedit-3.2', 'jedit-4.0', 'log4j-1.0', 'log4j-1.1', 'poi-1.5',
    'prop-1', 'prop-5', 'synapse-1.1', 'velocity-1.6', 'xalan-2.6'))
plt.legend()

plt.tight_layout()
plt.show()
