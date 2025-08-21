#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <boost/program_options.hpp>
#include "../pq/ivf_pq.h"
#include "../../include/utils.h"

namespace po = boost::program_options;



int main(int argc, char** argv) {
    std::string data_type, dist_fn, base_bin_file, query_bin_file, train_bin_file, base_label_file, query_label_file, train_label_file, gt_file, index_path_prefix;
    ANNS::IdxType K, Dim, Nprobe, Nlist, M_val, Nbits;

    try {
        po::options_description desc{"Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), 
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(), 
                           "distance function <L2/IP/cosine>");
        desc.add_options()("ivf_base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                           "File containing the base vectors in binary format");
        desc.add_options()("train_bin_file", po::value<std::string>(&train_bin_file)->required(),
                           "File containing the training vectors in binary format");
        desc.add_options()("query_bin_file", po::value<std::string>(&query_bin_file)->required(),
                           "File containing the query vectors in binary format");
        desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->default_value(""),
                           "Base label file in txt format");
        desc.add_options()("train_label_file", po::value<std::string>(&train_label_file)->default_value(""),
                           "Train label file in txt format");
        desc.add_options()("query_label_file", po::value<std::string>(&query_label_file)->default_value(""),
                           "Query label file in txt format");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                           "Filename for the writing ground truth in binary format");
        desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(),
                           "Number of ground truth nearest neighbors to compute");

        //ivf_pq parameters
        desc.add_options()("nprobe", po::value<ANNS::IdxType>(&Nprobe)->required(),
                           "Number of clusters to look through per query");
        desc.add_options()("nlist", po::value<ANNS::IdxType>(&Nlist)->required(),
                           "Number of clusters to create");
        desc.add_options()("dim", po::value<ANNS::IdxType>(&Dim)->required(),
                           "Number of dimensions");

        desc.add_options()("nbits", po::value<ANNS::IdxType>(&Nbits)->required(),
                           "Number of bits");
        desc.add_options()("m_val", po::value<ANNS::IdxType>(&M_val)->required(),
                           "m value");
    
                           
        


        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    // load base and query data
    std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
    std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
    std::shared_ptr<ANNS::IStorage> train_storage = ANNS::create_storage(data_type);
    base_storage->load_from_file(base_bin_file, base_label_file);
    query_storage->load_from_file(query_bin_file, query_label_file);
    train_storage->load_from_file(train_bin_file, train_label_file); 


    // load index
    IndexIVFPQ my_index(Dim, Nprobe, Nlist, Nbits, M_val);
    my_index.train(train_storage);
    my_index.add(base_storage);

    // perform queries 
    auto num_queries = query_storage->get_num_points(); 
    auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];
    auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
    ANNS::load_gt_file(gt_file, gt, num_queries, K);
    
    std::cout << "Start querying ..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    my_index.query(query_storage, K, results);
    auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
    

    std::cout << "- Time cost: " << time_cost << "ms" << std::endl;
    std::cout << "- QPS: " << num_queries * 1000.0 / time_cost << std::endl;


    // calculate recall
    auto recall = ANNS::calculate_recall(gt, results, num_queries, K);
    std::cout << "- Recall: " << recall << "%" << std::endl;
    return 0;
}