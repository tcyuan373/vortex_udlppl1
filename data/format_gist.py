from balanced_ann import *
from faiss_knn import *
import argparse

def main():
     parser = argparse.ArgumentParser(description='Run Cluster Partition command with custom directories.')
     parser.add_argument('--embeddings_loc', type=str, default='./perf_data/gist', help='Directory to save embeddings and related files(default: ./gist)')     parser.add_argument('--ncentroids', type=int, default=3, help='Number of centroids for KMeans clustering(default: 3)')
     parser.add_argument('-b', action='store_true', help='Enable the b flag')
     # gp_ann arguments
     parser.add_argument('--gp_ann_loc', type=str, default="./gp-ann" , help='Directory containing the compiled gp_ann code')
     # faiss arguments
     parser.add_argument('--niter', type=int, default=20, help='Number of iterations for KMeans clustering(default: 20)')
     args = parser.parse_args()
     use_balanced_knn = False
     if args.b:
          use_balanced_knn = True

     contain_all_files = check_and_clean_folder(args.embeddings_loc)
     if not contain_all_files:
          sys.exit(1)
     if use_balanced_knn:
          gp_ann_cluster(args)
     else:
          faiss_knn_cluster(args)


if __name__ == '__main__':
     main()