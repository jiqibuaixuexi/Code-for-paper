import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
# csv path
########################
parser.add_argument('--csv_path', type=str, default='val_acc.csv', help='csv path')
parser.add_argument('--metric', type=str, default='Accuracy', help='Metric')
########################
args = parser.parse_args()


data = pd.read_csv(args.csv_path)
print(data)
print(data.columns)
data = data.loc[:,['upperbound', 'SRC-MT']]
print(data.columns)
fig, ax = plt.subplots()
ax.plot(data.index, data['upperbound'], linestyle='-', label='upperbound')
ax.plot(data.index, data['SRC-MT'], linestyle='-', label='SRC-MT')
ax.set_xlabel('Observation Point') 
ax.set_ylabel(args.metric) 
ax.legend()
plt.show()
