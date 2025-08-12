#ifndef IVFFLAT_H
#define IVFFLAT_H

#include <iostream> 
#include <vector> 
#include "../../include/storage.h"

using namespace std; 

class IndexIVFFlat {
    public: 
        IndexIVFFlat(int d, int np, int nl); 
        void IndexIVFFlat::train(std::shared_ptr<IStorage> dataset);
        void IndexIVFFlat::add(std::shared_ptr<IStorage> dataset); 
        void IndexIVFFlat::query(std::shared_ptr<IStorage> dataset, int k, std::pair<IdxType, float>* results); 

    private: 

        float IndexIVFFlat::euclideanDistance(const char* a, const char* b); 
        vector<float> flattenDataset(const vector<vector<float>>& dataset);
        vector<vector<float>> convertToVectorOfVectors(const float* centroids, int k, int d);
        int dim; 
        int nprobe; 
        int nlist; 
        std::shared_ptr<IStorage> base_storage;
        vector<vector<float>> centroids; 
        vector<vector<int>> inverted_list; 

}; 


#endif