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
        
    def load_searcher_gpu(self):
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=8, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )
    
    def ocdpo_handler(self,**kwargs):
        blob                = kwargs["blob"]
        bytes_obj           = blob.tobytes()
        json_str_decoded    = bytes_obj.decode('utf-8')
        cluster_result      = json.loads(json_str_decoded)
        queries             = cluster_result['queries']
        query_embeds        = cluster_result['Qembeddings']
        
            
        if self.searcher == None:
            self.load_searcher_gpu()
            
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            query_embeddings=torch.Tensor(query_embeds),
            num_document_to_retrieve=1, # how many documents to retrieve for each query
            centroid_search_batch_size=1,
        )
        
        print('==========Finished Searching==========')
        print(f'Got a ranking dictionary: {ranking.todict()}') 
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass