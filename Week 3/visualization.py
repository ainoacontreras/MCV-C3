import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA

def reduce(data, mode):
    if mode == 'TSNE':
        reduced_data = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3).fit_transform(data)
    
    elif mode=='UMAP':
        reduced_data = umap.UMAP(n_components=2, n_neighbors=2).fit_transform(data)
    
    elif mode=='PCA':
        reduced_data = PCA(n_components=2).fit_transform(data)
    
    else:
        raise ValueError('Invalid mode. Please use one of the following: TSNE, UMAP, PCA')

    return reduced_data
    

embedding = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
mode = 'TSNE'
reduced_embedding = reduce(embedding, mode)
print(reduced_embedding)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1])
plt.title(f'Visualization of Embeddings using {mode}')
plt.xlabel(f'{mode}Dimension 1')
plt.ylabel(f'{mode}Dimension 2')
plt.show()