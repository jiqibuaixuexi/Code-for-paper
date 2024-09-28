
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib


parser = argparse.ArgumentParser()
# csv path
parser.add_argument('--csv_path', type=str, default='poolformer_SRC_MT_0.1_activations_result.csv', help='csv of activation results')

###################################################################################

###################################################################################

args = parser.parse_args()


data = pd.read_csv(args.csv_path)
print(data.shape)
# print('data\n',data)

labels = data.iloc[:,37632] 
print('labels\n',labels)
features = data.iloc[:,:37632]
print('features shape\n',features.shape)
# print('labels\n',labels)

model = TSNE(n_components=2)
tsne_features = model.fit_transform(features)


df = pd.DataFrame(tsne_features, columns=['x','y'])
y = labels
print(df.shape)



colors = ['#1f77b4', '#ff7f0e']
label_list = ['AS','nonAS']

for i in range(df.shape[1]):
    plt.scatter(df.loc[y==i,'x'], df.loc[y==i,'y'], c=colors[i], label=label_list[i],cmap=matplotlib.colors.ListedColormap(colors))

plt.legend()

plt.show()