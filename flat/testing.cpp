#include <faiss/IndexFlat.h>
#include "ivf_flat.h"
#include <vector>
#include <iostream>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <cassert>

using namespace std;

// --------- Your Custom IVF Index Header ---------
// class IndexIVFFlat {
// public:
//     IndexIVFFlat(int d, int nprobe, int nlist);
//     void train(vector<vector<float>> dataset);
//     void add(vector<vector<float>> dataset);
//     vector<vector<int>> query(vector<vector<float>> dataset, int k);
// };

float l2_distance(const float* a, const float* b, int d) {
    float sum = 0;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float compute_recall(
    const vector<vector<int>>& ground_truth,
    const vector<vector<int>>& predicted,
    int k
) {
    assert(ground_truth.size() == predicted.size());

    int total = 0;
    int hit = 0;

    for (size_t i = 0; i < ground_truth.size(); i++) {
        unordered_set<int> gt_set(ground_truth[i].begin(), ground_truth[i].end());
        for (int pid : predicted[i]) {
            if (gt_set.count(pid)) {
                hit++;
            }
        }
        total += k;
    }

    return static_cast<float>(hit) / total;
}

// Helper to flatten a 2D vector into raw float* for FAISS
void flatten(const vector<vector<float>>& input, float* output) {
    size_t idx = 0;
    for (const auto& vec : input) {
        for (float f : vec) {
            output[idx++] = f;
        }
    }
}

int main() {
    const int d = 64;         // vector dimensionality
    const int nb = 10000;     // database size
    const int nq = 100;       // number of queries
    const int k = 10;         // top-k
    const int nlist = 100;    // for your IVF
    const int nprobe = 10;

    // ---------- Generate Random Data ----------
    mt19937 rng(123);
    normal_distribution<float> dist;

    vector<vector<float>> xb(nb, vector<float>(d));
    vector<vector<float>> xq(nq, vector<float>(d));

    for (auto& vec : xb)
        for (auto& x : vec)
            x = dist(rng);

    for (auto& vec : xq)
        for (auto& x : vec)
            x = dist(rng);

    // ---------- Ground Truth with FAISS IndexFlatL2 ----------
    faiss::IndexFlatL2 index_flat(d);
    {
        vector<float> flat_xb(nb * d);
        flatten(xb, flat_xb.data());
        index_flat.add(nb, flat_xb.data());
    }

    vector<vector<int>> gt(nq); // ground truth
    {
        vector<float> flat_xq(nq * d);
        flatten(xq, flat_xq.data());

        vector<faiss::idx_t> I(nq * k);
        vector<float> D(nq * k);

        index_flat.search(nq, flat_xq.data(), k, D.data(), I.data());

        for (int i = 0; i < nq; i++) {
            gt[i].assign(I.begin() + i * k, I.begin() + (i + 1) * k);
        }
    }

    // ---------- Your IVF Index ----------
    IndexIVFFlat my_index(d, nprobe, nlist);
    my_index.train(xb);
    my_index.add(xb);
    vector<vector<int>> approx = my_index.query(xq, k);

    // ---------- Compute Recall ----------
    float recall = compute_recall(gt, approx, k);
    cout << "Recall@k = " << recall << endl;

    return 0;
}
