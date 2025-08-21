#include "ivf_flat.h"
#include "../../include/distance.h"
#include <cmath> 
#include <random>
#include <algorithm>  // for std::shuffle
#include <queue> 
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>  // needed as a temporary quantizer

using namespace std;
using namespace ANNS;  


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



void IndexIVFFlat::train(std::shared_ptr<IStorage> dataset) {
    //perform k means clustering 

    auto float_storage = std::dynamic_pointer_cast<ANNS::Storage<float>>(dataset);

    faiss::ClusteringParameters cp; 
    cp.verbose = false; 
    cp.niter = 20; 
    faiss::Clustering clus(dim, nlist, cp);
    faiss::IndexFlatL2 quantizer(dim);
    vector<float> data; 
    data.reserve(float_storage->get_num_points()); 
    for(int i = 0; i < float_storage->get_num_points(); i++) {
        float* v = reinterpret_cast<float*>(float_storage->get_vector(i));
        data.insert(data.end(), v, v + dim);
    }
    
    clus.train(float_storage->get_num_points(), data.data(), quantizer);


    centroids = convertToVectorOfVectors(clus.centroids.data(), nlist, dim); 
    std::cout << "centroids" << centroids.size() << std::endl;

}


void IndexIVFFlat::add(std::shared_ptr<IStorage> dataset) {
    auto float_storage = std::dynamic_pointer_cast<ANNS::Storage<float>>(dataset);
    base_storage.reserve(float_storage->get_num_points());
    for(int i = 0; i < float_storage->get_num_points(); i++) {
        int best_index = 0; 
        float* v = reinterpret_cast<float*>(float_storage->get_vector(i));
        float best_distance = euclideanDistance(reinterpret_cast<const char *>(v), reinterpret_cast<const char *>(centroids[0].data())); 
        for(int j = 1; j < centroids.size(); j++) {
            float distance = euclideanDistance(reinterpret_cast<const char *>(v), reinterpret_cast<const char *>(centroids[j].data())); 
            if(distance < best_distance) {
                best_distance = distance; 
                best_index = j; 
            }
        }

        base_storage.emplace_back(v, v + dim); 

 

        inverted_list[best_index].push_back(i); 
    }
}


using Pair = std::pair<float, int>;

struct Compare {
    bool operator()(const Pair& a, const Pair& b) {
        return a.first > b.first;  // Smallest value first
    }
};

void IndexIVFFlat::query(std::shared_ptr<IStorage> dataset, int k, std::pair<IdxType, float>* results) {
    auto float_storage = std::dynamic_pointer_cast<ANNS::Storage<float>>(dataset);
    std::pair<IdxType, float>* _results = results; 
    for(int i = 0; i < float_storage->get_num_points(); i++) {
        std::priority_queue<Pair, std::vector<Pair>, Compare> pq;
        for(int j = 0; j < centroids.size(); j++) {
            pq.push(std::make_pair(
                euclideanDistance(reinterpret_cast<const char *>(centroids[j].data()),
                    float_storage->get_vector(i)),
                j));
        }

        std::priority_queue<Pair, std::vector<Pair>, Compare> actual_vectors;

        for(int j = 0; j < nprobe; j++) {
            auto [distance, index] = pq.top(); 
            pq.pop(); 
            vector<int> centroid_vectors = inverted_list[index]; 
            for (int indexes : centroid_vectors) {
                actual_vectors.push(
                    std::pair<float, int>(
                    euclideanDistance(reinterpret_cast<const char *>(base_storage[indexes].data()),
                        float_storage->get_vector(i)),
                        indexes)
                    );
            }
        }



        for(int j = 0; j < k; j++) {
            auto [distance, index] = actual_vectors.top(); 
            actual_vectors.pop(); 
            _results[i * k + j] = { index, distance }; 
        }
    }

}


float IndexIVFFlat::euclideanDistance(const char* a, const char* b) {
    ANNS::FloatL2DistanceHandler distance_handler; 

    float dist = distance_handler.compute(
        a,
        b,
        dim
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






