import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
import pickle
import matplotlib.cm as cm


def open_file(file_path):
    file = open(file_path, "rb")
    embeddings, labels = pickle.load(file)
    file.close()
    return embeddings, labels


def reduce(data, mode):
    if mode == 'TSNE':
        reduced_data = TSNE(n_components=2, learning_rate='auto',
                    init='pca', perplexity=3).fit_transform(data)
    
    elif mode=='UMAP':
        reduced_data = umap.UMAP(n_components=2, n_neighbors=5).fit_transform(data)
    
    elif mode=='PCA':
        reduced_data = PCA(n_components=2).fit_transform(data)
    
    else:
        raise ValueError('Invalid mode. Please use one of the following: TSNE, UMAP, PCA')

    return reduced_data

file = './embeddings/resnet_validation_features.pkl'
embeddings, labels = open_file(file)

mode = 'UMAP'
reduced_embedding = reduce(embeddings, mode)

classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
colormap = cm.get_cmap('tab10')
class_colors = [colormap(i/len(classes)) for i in range(len(classes))]

plt.figure(figsize=(10, 8))
for label in range(len(classes)):
    mask = np.array([l == label for l in labels])
    plt.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], color=class_colors[label], label=classes[label])

plt.title(f'Visualization of Embeddings using {mode}')
plt.xlabel(f'{mode} Dimension 1')
plt.ylabel(f'{mode} Dimension 2')
plt.legend()
plt.show()