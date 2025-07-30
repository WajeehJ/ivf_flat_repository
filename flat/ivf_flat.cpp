#include "ivf_flat.h"
#include "../../include/distance.h"
#include <cmath> 
#include <random>
#include <algorithm>  // for std::shuffle
#include <queue> 
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>  // needed as a temporary quantizer

using namespace std; 


IndexIVFFlat::IndexIVFFlat(int d, int np, int nl) : dim(d), nprobe(np), nlist(nl) {
    inverted_list.resize(nlist); 
    centroids.resize(nlist); 
}



void printVector(const vector<float>& vec) {
    cout << "[ ";
    for (float val : vec) {
        cout << val << " ";
    }
    cout << "]" << endl;
}



void IndexIVFFlat::train(vector<vector<float>> dataset) {
    //perform k means clustering 
    faiss::ClusteringParameters cp; 
    cp.verbose = false; 
    cp.niter = 20; 
    faiss::Clustering clus(dim, nlist, cp);
    faiss::IndexFlatL2 quantizer(dim);
    clus.train(nlist, flattenDataset(dataset).data(), quantizer);

    centroids = convertToVectorOfVectors(clus.centroids.data(), nlist, dim); 

}


void IndexIVFFlat::add(vector<vector<float>> dataset) {

    database = dataset; 
    for(int i = 0; i < dataset.size(); i++) {
        int best_index = 0; 
        float best_distance = euclideanDistance(dataset[i], centroids[0]); 
        for(int j = 1; j < centroids.size(); j++) {
            float distance = euclideanDistance(dataset[i], centroids[j]); 
            if(distance < best_distance) {
                best_distance = distance; 
                best_index = j; 
            }
        }

        inverted_list[best_index].push_back(i); 
    }
}


using Pair = std::pair<float, int>;

struct Compare {
    bool operator()(const Pair& a, const Pair& b) {
        return a.first > b.first;  // Smallest value first
    }
};

vector<vector<int>> IndexIVFFlat::query(vector<vector<float>> dataset, int k) {
    vector<vector<int>> results; 
    results.resize(dataset.size()); 
    for(int i = 0; i < dataset.size(); i++) {
        std::priority_queue<Pair, std::vector<Pair>, Compare> pq;
        for(int j = 0; j < centroids.size(); j++) {
            pq.push({euclideanDistance(centroids.at(j), dataset.at(i)), i}); 
        }

        std::priority_queue<Pair, std::vector<Pair>, Compare> actual_vectors;

        for(int j = 0; j < nprobe; j++) {
            auto [distance, index] = pq.top(); 
            pq.pop(); 
            vector<int> centroid_vectors = inverted_list[index]; 
            for(int indexes : centroid_vectors) {
                actual_vectors.push({euclideanDistance(database[indexes], dataset[i]), indexes}); 
            }
        }



        vector<int> result; 
        result.resize(k); 
        for(int j = 0; j < k; j++) {
            auto [distance, index] = pq.top(); 
            pq.pop(); 
            result[j] = index; 
        }

        results[i] = result; 
    }

    return results; 
}


float IndexIVFFlat::euclideanDistance(const vector<float>& a, const vector<float>& b) {
    ANNS::FloatL2DistanceHandler distance_handler; 

    float dist = distance_handler.compute(
        reinterpret_cast<const char *>(a.data()),
        reinterpret_cast<const char *>(b.data()),
        a.size()
    );

    return dist; 
}


vector<vector<float>> IndexIVFFlat::convertToVectorOfVectors(const float* centroids, int k, int d) {
    vector<vector<float>> result(k, vector<float>(d));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            result[i][j] = centroids[i * d + j];
        }
    }

    return result;
}


vector<float> IndexIVFFlat::flattenDataset(const vector<vector<float>>& dataset) {
    if (dataset.empty()) return {};

    int num_points = dataset.size();
    int dim = dataset[0].size();

    vector<float> flat(num_points * dim);

    for (int i = 0; i < num_points; ++i) {
        // Optional: check that all rows have same dim
        for (int j = 0; j < dim; ++j) {
            flat[i * dim + j] = dataset[i][j];
        }
    }
    return flat;
}






