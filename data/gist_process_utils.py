import numpy as np
import os 
import pickle
from collections import defaultdict

def check_and_clean_folder(data_dir):
    # Check if 'gist_base.fvecs' exists in the directory
    gist_files = [ 'gist_base.fvecs', \
                'gist_groundtruth.ivecs', \
                'gist_query.fvecs',\
                'gist_learn.fvecs']
    for file in gist_files:
        file_pathname = os.path.join(data_dir, file)
        if not os.path.isfile(file_pathname):
            print(f"Error: '{file_pathname}' not found in {data_dir}.")
            return False
    # Clean the directory of all files except '.fvec' file
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path) and not filename in gist_files:
            os.remove(file_path)  
            print(f"Removed: {file_path}")
    return True

def fvecs_read(filename, dtype=np.float32, c_contiguous=True):
     fv = np.fromfile(filename, dtype=dtype)
     if fv.size == 0:
          return np.zeros((0, 0))
     dim = fv.view(np.int32)[0]
     assert dim > 0
     fv = fv.reshape(-1, 1 + dim)
     if not all(fv.view(np.int32)[:, 0] == dim):
          raise IOError("Non-uniform vector sizes in " + filename)
     fv = fv[:, 1:]
     if c_contiguous:
          fv = fv.copy()
     return fv


def write_query_results(data_dir):
     groundtruth_path = os.path.join(data_dir, 'gist_groundtruth.ivecs')
     groundtruth = fvecs_read(groundtruth_path, np.int32)
     print("groundtruth shape",groundtruth.shape)

     query_path = os.path.join(data_dir, 'gist_query.fvecs')
     query = fvecs_read(query_path)
     print("query shape", query.shape)

     # synthetic query texts
     querytexts = []
     for i in range(len(query)):
          querytexts.append("Query " + str(i))

     ground_truth_pathname = os.path.join(data_dir, 'groundtruth.csv')
     np.savetxt(ground_truth_pathname, groundtruth, delimiter=",", fmt='%i')

     query_pathname = os.path.join(data_dir, 'query.csv')
     np.savetxt(query_pathname, querytexts, fmt="%s")

     query_emb_pathname = os.path.join(data_dir, 'query_emb.csv')
     np.savetxt(query_emb_pathname, query, delimiter=",")


def write_cluster_embeddings(I, embs, ncentroids, data_dir):
     doc_emb_map = defaultdict(dict)
     clustered_embs = [[] for _ in range(ncentroids)]
     for i in range(len(embs)):
          cluster = I[i][0]
          if len(I[i]) != 1:
               print(f"Error in embedding {i}, len {len(I[i])}")
          clustered_embs[cluster].append(embs[i])
          emb_id = len(clustered_embs[cluster]) - 1
          doc_emb_map[cluster][emb_id] = i

     for i in range(ncentroids):
          pathname = os.path.join(data_dir, f'cluster_{i}.pkl')
          with open(pathname, 'wb') as f:
               pickle.dump(clustered_embs[i], f)
          print(f"Cluster {i} saved to {pathname} with {len(clustered_embs[i])} points.")

     doc_emb_map_path = os.path.join(data_dir, 'doc_emb_map.pkl')
     with open(doc_emb_map_path, 'wb') as f:
          pickle.dump(doc_emb_map, f)


