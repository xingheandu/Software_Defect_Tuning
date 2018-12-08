import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 14
sequential = (
    6.67, 7.28, 8.73, 11.32, 8.12, 8.24, 6.74, 6.59, 8.06, 154.08, 66.50, 7.48, 7.60, 14.41)
master_slave = (0.70, 0.73, 0.92, 1.23, 0.76, 0.87, 0.66, 0.62, 0.83, 19.39, 7.95, 0.71, 0.83, 1.45)

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
plt.title('Tuning Time of CART')
plt.xticks(rotation=90)
plt.xticks(index + bar_width, (
    'ant-1.3', 'ant-1.4', 'ant-1.6', 'camel-1.2', 'jedit-3.2', 'jedit-4.0', 'log4j-1.0', 'log4j-1.1', 'poi-1.5',
    'prop-1', 'prop-5', 'synapse-1.1', 'velocity-1.6', 'xalan-2.6'))
plt.legend()

plt.tight_layout()
plt.show()
