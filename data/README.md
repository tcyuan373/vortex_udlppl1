# Construct Dataset and Prebuild Indices

## Dataset

### 1. Download the dataset
Download the gist dataset could simply be done via running the download script we provided, under the directory /data/perf_data

```./download_gist.sh```

This script creates a gist folder that contains the data files: gist_base.fvecs,  gist_groundtruth.ivecs, gist_learn.fvecs, gist_query.fvecs

### 2. Clustering
There are two ways to build clusters, one is use FAISS KNN algorithm which could generate specified number of clusters, another way is to use balanced_knn algorithm from paper https://arxiv.org/abs/2403.01797 by Lars Gottesbüren, et al. We imported their codebase with customization for better alignment with our setup script

#### 2.1 FAISS KNN clustering
To use FAISS KNN clustering algorithm, you can simple run

``` python format_gist.py --embeddings_loc /path/to/save/embeddings --ncentroids 5 --niter 20```

#### 2.2 Use balanced KNN clustering
To get balanced knn clustering datasets, first needs to compile and build the gp-ann repository. (git submodule update --init --recursive
)
It could be built either via cmake in that directory (Note make sure to build the executables in folder named ```release_l2```) or the python file 
``` python build_balanced_gpann.py ```

To run gp-ann to generate balanced clustering, you can run with -b flag

``` python format_gist.py -b --embeddings_loc /path/to/save/embeddings --ncentroids 5 --gp_ann_loc ./gp-ann```

## Prebuild Indices for HNSW
Building HNSW indices take a long time. Vortex allows one to load prebuilt indices. The source code is under the `data/hnsw/` directory

**Building Indicies**

1. Locate the dataset folder under `perf_data/[dataset_name]`. The dataset folder should container a `centroids.pkl` file and multiple `cluster_*.pkl` files.
2. If those two files do not exist as in the case of the gist dataset, run `format_gist.py`
3. The built executable for hnsw_index is at `{build_directory}/data/build_hnsw_index`. It takes dataset directory via arguments `./build_hnsw_index {embedding_dir} {index_dir to store the prebuild indecies} -m {hnsw_m} -e {ef_construction}`. After cd into `{build_directory}/data/` directory, for exmample, if the dataset is called miniset, then the build command will look like `./build_hnsw_index perf_data/miniset perf_data/hnsw_index/miniset -m 100 200 -e 200 500`


**Loading Indicies**

To configure the cluster search udl to load the correct dataset when using hnsw, configure the `faiss_search_type` to 3 and `dataset_name` to match the name of the folder in benchmark/hnsw_index.

```
"user_defined_logic_config_list": [
{
      "emb_dim":1024,
      "top_k":3,
      "faiss_search_type":3,
      "dataset_name": "perf_data/hnsw_index/miniset", // folder containing the prebuilt indicies
      "hnsw_m": 100,                 // hnsw graph connectedness
      "hnsw_ef_construction": 200,  // hnsw exploration factor for construction
      "hnsw_ef_search:" 100         // hnsw exploration factor for search
}],
```

Note: the code looks at M, EF_CONSTRUCTION, and EF_SEARCH in `grouped_embeddings_for_search.hpp` to try and load the correct prebuilt index.
If it could find the corresponding prebuild hnsw bin with matching hnsw_m and hnsw_ef_construction, then it will load and use the prebuild indecies, otherwise, it will construct the indecies on-the-fly. 