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
    // bool finished = false;
    // int changedCentroids = 0; 
    
    // vector<int> indices = pickRandomIndices(nlist, nlist); 

    // //choose nlist random vectors from learning set to be centroids
    // for(int i = 0; i < nlist; i++) {
    //     centroids.at(i) = dataset.at(indices[i]); 
    // }

    // while(!finished) {

    //     //assign vectors to different centroids 
    //     for(int i = 0; i < dataset.size(); i++) {
    //         int best_index = 0; 
    //         float best_distance = euclideanDistance(dataset[i], centroids[0]); 
    //         for(int j = 1; j < centroids.size(); j++) {
    //             float distance = euclideanDistance(dataset[i], centroids[j]); 
    //             if(distance < best_distance) {
    //                 best_distance = distance; 
    //                 best_index = j; 
    //             }
    //         }

    //         inverted_list[best_index].push_back(i); 
    //     }


    //     //recalculate new centroids 
    //     for(int i = 0; i < inverted_list.size(); i++) {
    //         vector<float> new_centroid;
    //         new_centroid.resize(dim);  
    //         vector<int> current_list = inverted_list[i]; 
    //         if(current_list.empty()) { continue;  }
    //         for(int curr_vector_index : current_list) {
    //             vector<float> curr_vector = dataset[curr_vector_index];
    //             for(int j = 0; j < dim; j++) {
    //                 new_centroid[j] = curr_vector[j] + new_centroid[j]; 
    //             }
    //         }

    //         for(int j = 0; j < dim; j++) {
    //             new_centroid[j] /= current_list.size(); 
    //         }

    //         float diff = euclideanDistance(new_centroid, centroids[i]);
    //         if (diff > 1e-3f) { // You can adjust the threshold as needed
    //             changedCentroids++;
    //             centroids[i] = new_centroid;
    //         }
            
    //     }

    //     if(changedCentroids == 0) {
    //         finished = true; 
    //     }

    //     changedCentroids = 0; 

    //     for (auto& list : inverted_list) {
    //         list.clear();
    //     }


    // }


    //perform k means clustering 
    faiss::ClusteringParameters cp; 
    cp.verbose = false; 
    cp.niter = 20; 
    faiss::Clustering clus(dim, nlist, cp);
    faiss::IndexFlatL2 quantizer(dim);
    clus.train(nlist, flattenDataset(dataset).data(), quantizer);

    centroids = convertToVectorOfVectors(clus.centroids.data(), nlist, dim); 

    //clear out training vectors 
    // for(int i = 0; i < inverted_list.size(); i++) {
    //     inverted_list[i].clear(); 
    // }
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

vector<int> IndexIVFFlat::pickRandomIndices(int size, int n) {
    // Create a vector with numbers 0, 1, 2, ..., size-1
    std::vector<int> indices(size);
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }

    // Use a random device to seed the generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Shuffle the vector randomly
    std::shuffle(indices.begin(), indices.end(), gen);

    // Take the first n elements after shuffle
    if (n > size) n = size;  // Clamp n to size
    indices.resize(n);

    return indices;
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






