import numpy as np
import subprocess
import os
import struct
import sys
from collections import Counter
import pickle
from gist_process_utils import *



def fvecs_to_fbin_with_metadata(gist_data_directory):
    input_file = os.path.join(gist_data_directory, 'gist_base.fvecs')
    output_file = os.path.join(gist_data_directory, 'gist_base.fbin')
    with open(input_file, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    dim = int(data.view(np.int32)[0])
    assert dim > 0, "Invalid dimension read from file."
    
    data = data.reshape(-1, 1 + dim)
    if not np.all(data[:, 0].view(np.int32) == dim):
        raise IOError(f"Non-uniform vector sizes in {input_file}")
    
    n = data.shape[0]
    vectors = data[:, 1:]
    with open(output_file, "wb") as f:
        f.write(np.array([n, dim], dtype=np.uint32).tobytes())  # Write n and d
        vectors.tofile(f)
    print(f"Converted {input_file} to {output_file} with metadata (n={n}, d={dim})")



def run_balanced_knn(gp_ann_directory, gist_directory, num_clusters):
    gist_base_fbin = os.path.join(gist_directory, 'gist_base.fbin')
    gist_partition = os.path.join(gist_directory, 'gist.partition')
    partition_executable = os.path.join(gp_ann_directory, 'release_l2', 'Partition')
    
    command = [
        partition_executable, 
        gist_base_fbin, 
        gist_partition, 
        str(num_clusters),  
        'BalancedKMeans', 
        'default'
    ]
    
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for stdout_line in process.stdout:
            print(stdout_line, end='')  
        for stderr_line in process.stderr:
            print(stderr_line, end='')  

        return_code = process.wait()
        if return_code != 0:
            print(f"Error: Command returned non-zero exit code {return_code}")
        else:
            print("Command executed successfully.")



def load_partition(filepath):
    with open(filepath, 'rb') as f:
        n = struct.unpack('I', f.read(4))[0]  # 'I' means an unsigned 32-bit integer
        partition_result = struct.unpack(f'{n}i', f.read(n * 4))  # 'i' means signed 32-bit integers
        return partition_result, n


def read_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        embeddings = np.frombuffer(f.read(n * d * 4), dtype=np.float32)
        embeddings = embeddings.reshape((n, d))
        return embeddings, n, d


def gpann_write_cluster_embeddings(cluster_id, embeddings, partition_result, output_dir):
    
    cluster_filepath_dat = os.path.join(output_dir, f'cluster_{cluster_id}.dat')
    cluster_filepath_pkl = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
    # 1. Write cluster embeddings to files
    cluster_embeddings = embeddings[partition_result == cluster_id]
    n = cluster_embeddings.shape[0]
    # Write the selected embeddings to a binary .dat file
    with open(cluster_filepath_dat, 'wb') as f:
        n = cluster_embeddings.shape[0]
        d = cluster_embeddings.shape[1]
        f.write(np.uint32(n).tobytes())
        f.write(np.uint32(d).tobytes())
        f.write(cluster_embeddings.tobytes())
    print(f"Cluster {cluster_id} saved to {cluster_filepath_dat} with {n} points.")
    
    # Write the selected embeddings to a pickle .pkl file
    with open(cluster_filepath_pkl, 'wb') as f:
        pickle.dump(cluster_embeddings, f)
    print(f"Cluster {cluster_id} saved to {cluster_filepath_pkl} with {n} points.")

    # 2. Write cluster emb ID to embedding ID mapping
    doc_emb_map = defaultdict(dict)
    for emb_id, emb in enumerate(cluster_embeddings):
        doc_id = np.where(partition_result == cluster_id)[0][emb_id]
        doc_emb_map[cluster_id][emb_id] = doc_id

    doc_emb_map_path = os.path.join(output_dir, 'doc_emb_map.pkl')
    with open(doc_emb_map_path, 'wb') as f:
        pickle.dump(doc_emb_map, f)
    print(f"cluster embID to embedding ID map saved to {doc_emb_map_path}")

def gpann_process_cluster_results(data_dir, ncentroids):
    partition_filepath = os.path.join(data_dir, 'gist.partition.dat')
    embeddings_path = os.path.join(data_dir, 'gist_base.fbin')
    output_dir = data_dir
    # Load partition results
    partition_result, n = load_partition(partition_filepath)
    print(f"Loaded partition with {n} points.")
    # Load embeddings
    embeddings, _, _ = read_embeddings(embeddings_path)
    print(f"Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions.")
    # Write cluster embeddings to files
    I = np.array(partition_result).reshape(-1, 1)
    write_cluster_embeddings(I, embeddings, ncentroids, output_dir)
    # for centroid in centroid_counts.keys():
    #     gpann_write_cluster_embeddings(centroid, embeddings, partition_result, output_dir)


def load_centroids(filepath):
    with open(filepath, 'rb') as f:
        n, d = struct.unpack('2I', f.read(8))  
        centroids = []
        for _ in range(n):
            centroid = struct.unpack(f'{d}f', f.read(d * 4)) 
            centroids.append(centroid)
        return centroids, n, d


def save_centroids_to_pkl(centroids, filedir):
    filepath = os.path.join(filedir, "centroids.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(centroids, f)
    print(f"Centroids saved to {filepath}")


def gpann_process_centroid_results(data_dir):
    centroid_path = os.path.join(data_dir, "gist.partition_centroids.dat")
    centroids, n, d = load_centroids(centroid_path)
    print(f"Loaded {n} centroids with {d} dimensions.")
    save_centroids_to_pkl(centroids, data_dir)


def gp_ann_cluster(args):
    data_dir = args.embeddings_loc

    # 1. Convert gist_base.fvecs to gist_base.fbin
    fvecs_to_fbin_with_metadata(data_dir)
    # 2. Run Balanced KMeans
    run_balanced_knn(args.gp_ann_loc, data_dir, args.ncentroids)
    # 3. write results to pkls
    gpann_process_cluster_results(data_dir, args.ncentroids)
    gpann_process_centroid_results(data_dir)
    write_query_results(data_dir)


# def gp_ann_cluster():
#     parser = argparse.ArgumentParser(description='Run Partition command with custom directories.')
#     parser.add_argument('--gp_ann_loc', type=str, help='Directory containing the gp_ann code (e.g., ../gp_ann)')
#     parser.add_argument('--embeddings_loc', type=str, default='./gist', help='Directory to save embeddings and related files(default: ./gist)')
#     parser.add_argument('--ncentroids', type=int, default=3, help='Number of centroids for KMeans clustering(default: 3)')
#     args = parser.parse_args()

#     EMBEDDINGS_LOC = args.embeddings_loc
#     contain_all_files = check_and_clean_folder(EMBEDDINGS_LOC)
#     if not contain_all_files:
#         sys.exit(1)

#     # 1. Convert gist_base.fvecs to gist_base.fbin
#     fvecs_to_fbin_with_metadata(EMBEDDINGS_LOC)
#     # 2. Run Balanced KMeans
#     run_balanced_knn(args.gp_ann_loc, EMBEDDINGS_LOC, args.ncentroids)
#     # 3. write results to pkls
#     gpann_process_cluster_results(EMBEDDINGS_LOC, args.ncentroids)
#     gpann_process_centroid_results(EMBEDDINGS_LOC)
#     write_query_results(EMBEDDINGS_LOC)

# if __name__ == "__main__":

#     gp_ann_cluster()