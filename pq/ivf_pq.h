#ifndef IVFPQ_H
#define IVFPQ_H

#include <iostream> 
#include <vector> 

using namespace std; 

class IndexIVFPQ {
    public: 
        IndexIVFPQ(int d, int np, int nl, int b, int m); 
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
        int nbits; 
        int m_val; 
        vector<vector<float>> database; 
        vector<vector<float>> centroids; 
        vector<vector<int>> inverted_list; 

}; 


#endif