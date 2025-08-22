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
    codebooks.resize(m); 
}


void IndexIVFPQ::train(std::shared_ptr<IStorage> dataset) {
    //create coarse centroids 
    int k = pow(2, nbits); 

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

    vector<vector<vector<float>>> subspaces(m_val); 
    //create codebooks 


    //split every vector into M subspaces
    for (int i = 0; i < float_storage->get_num_points(); i++) {
        float* vec = reinterpret_cast<float*>(float_storage->get_vector(i)); 
        for (int m = 0; m < m_val; ++m) {

            int offset = m * (dim / m_val);
            vector<float> subvec(vec + offset, vec + offset + (dim / m_val));
            subspaces.at(m).push_back(subvec);
        }
    }


    //perform k means clustering on every subspace 
    for(int i = 0; i < subspaces.size(); i++) {
        vector<vector<float>> subspace = subspaces[i];
        vector<vector<float>> mth_centroid_list; 
        faiss::ClusteringParameters cp; 
        cp.verbose = false; 
        cp.niter = 20; 
        int sub_dim = subspace[0].size();  
        faiss::Clustering clus(sub_dim, k, cp);
        faiss::IndexFlatL2 quantizer(sub_dim);

        std::vector<float> flat_data = flattenDataset(subspace);
        clus.train(subspace.size(), flat_data.data(), quantizer);
        mth_centroid_list = convertToVectorOfVectors(clus.centroids.data(), nlist, sub_dim); 
        codebooks[i] = mth_centroid_list; 
    }
}


void IndexIVFPQ::add(std::shared_ptr<IStorage> dataset) {
    auto float_storage = std::dynamic_pointer_cast<ANNS::Storage<float>>(dataset);
    base_storage.reserve(float_storage->get_num_points());
    //assign every vector to a coarse vector
    for(int i = 0; i < float_storage->get_num_points(); i++) {
        int best_index = 0; 
        float* v = reinterpret_cast<float*>(float_storage->get_vector(i));
        float best_distance = euclideanDistance(reinterpret_cast<const char *>(v), reinterpret_cast<const char *>(centroids[0].data()), dim);
        for(int j = 1; j < centroids.size(); j++) {
            float distance = euclideanDistance(reinterpret_cast<const char *>(v), reinterpret_cast<const char *>(centroids[j].data()), dim);
            if(distance < best_distance) {
                best_distance = distance; 
                best_index = j; 
            }

        }


        //create compressed vector 
        vector<float> compressed_vector;

        for (int m = 0; m < m_val; ++m) {

            //create a subvector of D/M, now we need to push to corresponding m subspace
            int offset = m * (dim / m_val);
            vector<float> subvec(v + offset, v + offset + (dim / m_val));
            
            vector<vector<float>> mth_centroid_list = codebooks[m]; 

 

            int centroid_index = 0; 
            float best_centroid_distance = euclideanDistance(reinterpret_cast<const char *>(subvec.data()), reinterpret_cast<const char *>(mth_centroid_list[0].data()), dim / m_val); 
            for(int j = 0; j < mth_centroid_list.size(); j++) {
                float distance = euclideanDistance(reinterpret_cast<const char *>(subvec.data()), reinterpret_cast<const char *>(mth_centroid_list[j].data()), dim / m_val); 
                if(distance < best_centroid_distance) {
                    best_centroid_distance = distance; 
                    centroid_index = j; 
                }
            }

            compressed_vector.push_back(centroid_index); 
        }

        inverted_list[best_index].push_back(i);

        base_storage.emplace_back(compressed_vector); 
        
    }
}


using Pair = std::pair<float, int>;

struct Compare {
    bool operator()(const Pair& a, const Pair& b) {
        return a.first > b.first;  // Smallest value first
    }
};

void IndexIVFPQ::query(std::shared_ptr<IStorage> dataset, int k, std::pair<IdxType, float>* results) {
    auto float_storage = std::dynamic_pointer_cast<ANNS::Storage<float>>(dataset);
    std::pair<IdxType, float>* _results = results; 
    for(int i = 0; i < float_storage->get_num_points(); i++) {
        float* v = reinterpret_cast<float*>(float_storage->get_vector(i));
        std::priority_queue<Pair, std::vector<Pair>, Compare> pq;
        for(int j = 0; j < centroids.size(); j++) {
            pq.push({euclideanDistance(reinterpret_cast<const char *>(centroids.at(j).data()), reinterpret_cast<const char *>(v), dim), j}); 
        }

        std::priority_queue<Pair, std::vector<Pair>, Compare> actual_vectors;

        for(int j = 0; j < nprobe; j++) {
            auto [distance, index] = pq.top(); 
            pq.pop(); 
            vector<int> centroid_vectors = inverted_list[index]; 
            for(int indexes : centroid_vectors) {
                vector<float> compressed_vector = base_storage[indexes]; 
                float calculated_distance = 0; 

                for(int m = 0; m < m_val; m++) {
                    int offset = m * (dim / m_val);
                    vector<float> subvec(v + offset, v + offset + (dim / m_val));

                    calculated_distance += euclideanDistance(reinterpret_cast<const char *>(codebooks[m][compressed_vector[m]].data()), reinterpret_cast<const char *>(subvec.data()), dim / m_val); 
                }
                actual_vectors.push({calculated_distance, indexes}); 
            }
        }



        for(int j = 0; j < k; j++) {
            auto [distance, index] = actual_vectors.top(); 
            actual_vectors.pop(); 
            _results[i * k + j] = { index, distance }; 
        }
    }
}


float IndexIVFPQ::euclideanDistance(const char* a, const char* b, int dimension) {
    ANNS::FloatL2DistanceHandler distance_handler; 

    float dist = distance_handler.compute(
        a,
        b,
        dimension
    );

    return dist; 
}


vector<vector<float>> IndexIVFPQ::convertToVectorOfVectors(const float* centroids, int k, int d) {
    vector<vector<float>> result(k, vector<float>(d));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            result[i][j] = centroids[i * d + j];
        }
    }

    return result;
}


vector<float> IndexIVFPQ::flattenDataset(const vector<vector<float>>& dataset) {
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

