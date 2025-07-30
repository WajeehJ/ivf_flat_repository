#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cassert> 
#include <sys/stat.h>
#include <cstdio> 
#include <cstdlib> 
#include <cstring> 
#include "ivf_flat.h"
#include <algorithm>
#include <faiss/IndexFlat.h>
#include <random> 

using namespace std; 

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

std::vector<std::vector<float>> toVector2D(float* data, size_t d, size_t n) {
    std::vector<std::vector<float>> result(n, std::vector<float>(d));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            result[i][j] = data[i * d + j];
        }
    }
    return result;
}


std::vector<std::vector<int>> toVector2DInt(const int* data, size_t d, size_t n) {
    std::vector<std::vector<int>> result(n, std::vector<int>(d));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[i][j] = data[i * d + j];
        }
    }
    return result;
}



int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}



void printVector(const vector<int>& vec) {
    cout << "[ ";
    for (float val : vec) {
        cout << val << " ";
    }
    cout << "]" << endl;
}

void printFloatVector(const vector<float>& vec) {
    cout << "[ ";
    for (float val : vec) {
        cout << val << " ";
    }
    cout << "]" << endl;
}



int main() {
    // vector<vector<float>> base = toVector2D(fvecs_read("/users/wajeehj/pathfinder/data/sift/sift_base.fvecs", &d, &n), d, n);
    // vector<vector<float>> learn = toVector2D(fvecs_read("/users/wajeehj/pathfinder/data/sift/sift_learn.fvecs", &d, &n), d, n);
    // vector<vector<float>> query = toVector2D(fvecs_read("/users/wajeehj/pathfinder/data/sift/sift_query.fvecs", &d, &n), d, n);
    // vector<vector<int>> ground_truth = toVector2DInt(ivecs_read("/users/wajeehj/pathfinder/data/sift/sift_groundtruth.ivecs", &d, &n), d, n); 

    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            xb[d * i + j] = distrib(rng);
        }
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) {
            xq[d * i + j] = distrib(rng);
        }
        xq[d * i] += i / 1000.;
    }

    faiss::IndexFlatL2 index(d); // call constructor

    

    cout << "training the index" << endl; 
    IndexIVFFlat index(d, nprobe, nlist); 
    index.train(learn); 

    cout << "adding the data" << endl; 
    index.add(base); 

    cout << "performing queries" << endl; 
    vector<vector<int>> query_responses = index.query(query, k); 

    cout << "first query results" << endl; 

    printVector(query_responses.at(0)); 
    vector<int> first_10(ground_truth.at(0).begin(), ground_truth.at(0).begin() + 10); 
    printVector(first_10); 

    cout << "calculating recall" << endl; 

    float recall; 
    for(int i = 0; i < query_responses.size(); i++) {
        vector<int> vec1 = query_responses.at(i); 
        vector<int> vec2 = ground_truth.at(i); 

        size_t count = std::min<size_t>(k, std::min(vec1.size(), vec2.size()));

        std::vector<int> first10_vec1(vec1.begin(), vec1.begin() + count);
        std::vector<int> first10_vec2(vec2.begin(), vec2.begin() + count);


        std::sort(first10_vec1.begin(), first10_vec1.end());
        std::sort(first10_vec2.begin(), first10_vec2.end());

        std::vector<int> intersection;
        std::set_intersection(
            first10_vec1.begin(), first10_vec1.end(),
            first10_vec2.begin(), first10_vec2.end(),
            std::back_inserter(intersection)
        );

        recall += intersection.size(); 
    }

    recall /= (k * ground_truth.size()); 

    cout << "Recall " << endl; 
    cout << recall << endl;
    
}
