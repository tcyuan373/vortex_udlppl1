import torch
from easydict import EasyDict
import numpy as np
from transformers import AutoImageProcessor
from flmr import (
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRConfig,
)
from flmr import create_searcher, search_custom_collection

class MONOFLMR:
    
    def __init__(self, index_root_path, 
                 index_name, index_experiment_name, 
                 checkpoint_path, image_processor_name):
        self.index_root_path        = index_root_path
        self.index_name             = index_name
        self.index_experiment_name  = index_experiment_name
        self.checkpoint_path        = checkpoint_path
        self.image_processor_name   = image_processor_name
        self.Ks                     = [1]
        self.use_gpu                = True
        self.nbits                  = 8
        self.query_batch_size       = 8
        self.flmr_config            = None
        self.query_tokenizer        = None
        self.context_tokenizer      = None
        self.flmr_model             = None
        self.image_processor        = None
        self.searcher               = None
        
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                        text_config=self.flmr_config.text_config,
                                                                        subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                        text_config=self.flmr_config.text_config,
                                                                        subfolder="context_tokenizer")
        self.flmr_model = FLMRModelForRetrieval.from_pretrained(
            self.checkpoint_path,
            query_tokenizer=self.query_tokenizer,
            context_tokenizer=self.context_tokenizer,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.image_processor_name)
        
    def load_model_gpu(self):
        self.flmr_model = self.flmr_model.to("cuda")
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=self.nbits, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )
        
    def execFLMR(self,
                 input_ids,
                 attention_mask,
                 pixel_values,
                 question_ids,
                 text_sequence,
                 ):
        if self.flmr_model == None:
            self.load_model_cpu()
            
        if self.searcher == None:
            self.load_model_gpu()
        
        query_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        with torch.no_grad():
            query_embeddings = self.flmr_model.query(**query_input).late_interaction_output
        
        queries = dict(zip(question_ids, text_sequence))
        
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries= queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            centroid_search_batch_size=None,
        )
        
        return ranking.todict()