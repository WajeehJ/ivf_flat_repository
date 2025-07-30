#include "ivf_pq.h"
#include <cmath> 
#include "../../include/distance.h"
#include <random>
#include <algorithm>  // for std::shuffle
#include <queue> 
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>  // needed as a temporary quantizer

using namespace std; 

IndexIVFPQ::IndexIVFPQ(int d, int np, int nl, int b, int m) : dim(d), nprobe(np), nlist(nl), nbits(b), m_val(m) {
    inverted_list.resize(nlist); 
    centroids.resize(nlist); 
}


void IndexIVFPQ::train(vector<vector<float>> dataset) {
    //create coarse centroids 
    int k = pow(2, nbits); 

    faiss::ClusteringParameters cp; 
    cp.verbose = false; 
    cp.niter = 20; 
    faiss::Clustering clus(dim, nlist, cp);
    faiss::IndexFlatL2 quantizer(dim);
    clus.train(nlist, flattenDataset(dataset).data(), quantizer);

    centroids = convertToVectorOfVectors(clus.centroids.data(), nlist, dim); 

    vector<vector<vector<float>>> subspaces; 
    //create codebooks 

    //split every vector into M subspaces
    for (const auto& vec : dataset) {
        for (int m = 0; m < m_val; ++m) {

            //create a subvector of D/M, now we need to push to corresponding m subspace
            vector<float> subvec = vector<float>(
                vec.begin() + m * (dim / m_val),
                vec.begin() + (m + 1) * (dim / m_val)
            );

            subspaces.at(m).push_back(subvec); 

        }
    }

    //perform k means clustering on every subspace 
    for(const auto& subspace : subspaces) {

        vector<vector<float>> mth_centroid_list; 
        faiss::ClusteringParameters cp; 
        cp.verbose = false; 
        cp.niter = 20; 
        faiss::Clustering clus(dim, k, cp);
        faiss::IndexFlatL2 quantizer(dim);
        clus.train(nlist, flattenDataset(subspace).data(), quantizer);

        mth_centroid_list = convertToVectorOfVectors(clus.centroids.data(), nlist, dim); 
        
        codebooks.push_back(mth_centroid_list); 
    }
}


void IndexIVFPQ::add(vector<vector<float>> dataset) {
    database = dataset; 

    //assign every vector to a coarse vector
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


        //create compressed vector 

        for (int m = 0; m < m_val; ++m) {

            //create a subvector of D/M, now we need to push to corresponding m subspace
            vector<float> subvec = vector<float>(
                dataset[i].begin() + m * (dim / m_val),
                dataset[i].begin() + (m + 1) * (dim / m_val)
            );
            
            vector<vector<float>> mth_centroid_list = codebooks[m]; 

            vector<float> compressed_vector; 

            int centroid_index = 0; 
            float best_centroid_distance = euclideanDistance(subvec, mth_centroid_list[0]); 
            for(int j = 0; j < mth_centroid_list.size(); j++) {
                float distance = euclideanDistance(subvec, mth_centroid_list[j]); 
                if(distance < best_distance) {
                    best_centroid_distance = distance; 
                    centroid_index = j; 
                }
            }

            compressed_vector.push_back(centroid_index); 
        }

        inverted_list[best_index].push_back(i);
        
        database[i] = compressed_vector; 
    }
}


using Pair = std::pair<float, int>;

struct Compare {
    bool operator()(const Pair& a, const Pair& b) {
        return a.first > b.first;  // Smallest value first
    }
};

vector<vector<int>> IndexIVFPQ::query(vector<vector<float>> dataset, int k) {
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
                vector<float> compressed_vector = database[indexes]; 
                float distance = 0; 

                for(int m = 0; m < m_val; m++) {
                    distance += euclideanDistance(codebooks[m][compressed_vector[m]], dataset[i]); 
                }
                actual_vectors.push({distance, indexes}); 
            }
        }



        vector<int> result; 
        result.resize(k); 
        for(int j = 0; j < k; j++) {
            auto [distance, index] = actual_vectors.top(); 
            actual_vectors.pop(); 
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

