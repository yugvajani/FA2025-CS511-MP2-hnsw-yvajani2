import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query
    with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
       
        train_embeddings = f['train'][:]
        test_embeddings = f['test'][:]

    # Get dimensions
    d = train_embeddings.shape[1]
    M = 16

    # Creating HNSW index
    index = faiss.IndexHNSWFlat(d, M)

    index.hnsw.efConstruction = 200
    index.add(train_embeddings)
    index.hnsw.efSearch = 200

    query_vector = test_embeddings[0:1]
    k = 10
    
    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    distances, indices = index.search(query_vector, k)
    neighbor_indices = indices[0]

    print(f"Top 10 nearest neighbor indices: {neighbor_indices}")
    print(f"Distances: {distances[0]}")

    with open('output.txt', 'w') as f:
        for idx in neighbor_indices:
            f.write(f"{idx}\n")
    

if __name__ == "__main__":
    evaluate_hnsw()
