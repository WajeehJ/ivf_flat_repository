#ifndef IVFPQ_H
#define IVFPQ_H

#include <iostream> 
#include <vector> 
#include "../../include/storage.h"

using namespace std; 
using namespace ANNS; 

class IndexIVFPQ {
    public: 
        IndexIVFPQ(int d, int np, int nl, int b, int m); 
        void train(std::shared_ptr<IStorage> dataset);
        void add(std::shared_ptr<IStorage> dataset); 
        void query(std::shared_ptr<IStorage> dataset, int k, std::pair<IdxType, float>* results);

    private: 

        float euclideanDistance(const char* a, const char* b, int dimension); 
        vector<float> flattenDataset(const vector<vector<float>>& dataset);
        vector<vector<float>> convertToVectorOfVectors(const float* centroids, int k, int d);
        int dim; 
        int nprobe; 
        int nlist; 
        int nbits; 
        int m_val; 
        vector<vector<float>> base_storage; 
        vector<vector<float>> centroids; 
        vector<vector<int>> inverted_list; 
        vector<vector<vector<float>>> codebooks;

}; 


#endif