#ifndef IVFFLAT_H
#define IVFFLAT_H

#include <iostream> 
#include <vector> 

using namespace std; 

class IndexIVFFlat {
    public: 
        IndexIVFFlat(int d, int np, int nl); 
        void train(vector<vector<float>> dataset);
        void add(vector<vector<float>> dataset); 
        vector<vector<int>> query(vector<vector<float>> dataset, int k); 

    private: 

        float euclideanDistance(const vector<float>& a, const vector<float>& b); 
        vector<float> flattenDataset(const vector<vector<float>>& dataset);
        vector<vector<float>> convertToVectorOfVectors(const float* centroids, int k, int d);
        int dim; 
        int nprobe; 
        int nlist; 
        vector<vector<float>> database; 
        vector<vector<float>> centroids; 
        vector<vector<int>> inverted_list; 

}; 


#endif