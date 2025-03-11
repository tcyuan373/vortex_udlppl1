import torch
from flmr import (
    search_custom_collection, create_searcher
)
import os

class StepESearch:
    
    def __init__(self, index_root_path, index_experiment_name, index_name):
        self.searcher = None
        self.index_root_path        = index_root_path
        self.index_experiment_name  = index_experiment_name
        self.index_name             = index_name
        self.centroid_search_batch_size = None
        
        

    def load_searcher_gpu(self):

        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=8, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )
        
    def process_search(self, queries, query_embeddings):
        if self.searcher == None:
            self.load_searcher_gpu()
            
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            centroid_search_batch_size=self.centroid_search_batch_size
        )
        
        
        return ranking.todict()