import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import pickle
from network import NeuralNetwork

def tsne_visualization(model, words):
    embedding = model.network[0]
    X = np.array(embedding)
    X_rounded = np.round(X, decimals=1)

    results = TSNE(n_components=2).fit_transform(X_rounded)
    tsne_results = pd.DataFrame(results, columns=['tsne1', 'tsne2'])

    for i in range(250):
        plt.scatter(tsne_results['tsne1'][i], tsne_results['tsne2'][i], marker='x', color='red')
        plt.text(tsne_results['tsne1'][i], tsne_results['tsne2'][i], words[i], fontsize=9)

    plt.show()

def main():

    file = open("model.pk", 'rb')
    my_model = pickle.load(file)
    file.close()
    words = np.load('data/vocab.npy')

    # TSNE Visualization:
    tsne_visualization(my_model, words)


if __name__ == '__main__':
    main()