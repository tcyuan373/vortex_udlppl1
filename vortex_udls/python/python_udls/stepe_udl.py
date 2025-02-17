#!/usr/bin/env python3
import os
import numpy as np
import json
import re
import struct
import warnings
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
import torch
from torch import Tensor, nn
from flmr import (
    FLMRConfig, 
    FLMRQueryEncoderTokenizer, 
    FLMRContextEncoderTokenizer, 
    FLMRModelForRetrieval, 
    FLMRTextModel,
    search_custom_collection, create_searcher
)
import functools
import tqdm
import tqdm


class IntermediateResult:
    def __init__(self):
        self._queries           = None
        self._query_embeddings  = None
    
    def has_all(self):
        has_all = self._queries != None and self._query_embeddings != None
        return has_all


class StepEUDL(UserDefinedLogic):
    '''
    ConsolePrinter is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepEUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        print(f"ConsolePrinter constructor received json configuration: {self.conf}")

        self.searcher = None
        self.index_root_path        = './perf_data/pipeline1/index/'
        self.index_experiment_name  = 'test_experiment'
        self.index_name             = 'test_index'
        self.collected_intermediate_results = {}
        
    def load_searcher_gpu(self):
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=8, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )
    
    def process_search(self, queries, query_embeddings, bsize):
        if self.searcher == None:
            self.load_searcher_gpu()
            
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            query_embeddings=torch.Tensor(query_embeddings),
            num_document_to_retrieve=1, # how many documents to retrieve for each query
            centroid_search_batch_size=bsize,
        )
        
        
        return ranking.todict()
    
    
    def ocdpo_handler(self,**kwargs):
        key                 = kwargs["key"]
        blob                = kwargs["blob"]
        bytes_obj           = blob.tobytes()
        json_str_decoded    = bytes_obj.decode('utf-8')
        cluster_result      = json.loads(json_str_decoded)
        queries_texts       = cluster_result['queries']
        query_embeddings    = cluster_result['query_embeddings']
        bsize               = 32
        
        queries = {i: queries_texts[i] for i in range(len(queries_texts))}
        
        step_D_idx = key.find('stepD')
        uds_idx = key.find("_")
        batch_id = int(key[uds_idx+1:])
        
        if not self.collected_intermediate_results.get(batch_id):
            self.collected_intermediate_results[batch_id] = IntermediateResult()
            
        if step_D_idx != -1:
            self.collected_intermediate_results[batch_id]._queries = queries
            self.collected_intermediate_results[batch_id]._query_embeddings = query_embeddings
            
        if not self.collected_intermediate_results[batch_id].has_all():
            return
            
        ranking = self.process_search(self.collected_intermediate_results[batch_id]._queries, 
                                      self.collected_intermediate_results[batch_id]._query_embeddings,
                                      bsize)
        
        print('==========Finished Searching==========')
        print(f'Got a ranking dictionary: {ranking}')
        
        # erase the batch id dict{} 
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass