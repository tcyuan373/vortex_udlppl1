#!/usr/bin/env python3
import numpy as np
import json
import re
import struct
import warnings
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
import os
import torch
from torch import Tensor, nn
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel


def deserialize_string_list(serialized_data):
    """Deserialize a custom binary format into a list of strings."""
    num_elements = struct.unpack("I", serialized_data[:4])[0]  # Read the number of elements
    offset_size = num_elements * 4  # Each offset is 4 bytes

    if num_elements == 0:
        return []

    offsets = struct.unpack(f"{num_elements}I", serialized_data[4:4 + offset_size])  # Read offsets
    string_section = serialized_data[4 + offset_size:]  # Extract string section

    # Extract strings using offsets
    string_list = []
    for i in range(num_elements):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < num_elements else len(string_section)
        string_list.append(string_section[start:end].decode('utf-8'))

    return string_list


class StepAUDL(UserDefinedLogic):
    '''
    ConsolePrinter is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepAUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        self.checkpoint_path            = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_encoder_path         = 'perf_data/pipeline1/models/models_step_A_query_text_encoder.pt'
        self.local_projection_path      = 'perf_data/pipeline1/models/models_step_A_query_text_linear.pt'
        self.flmr_config                = None
        self.query_tokenizer            = None
        self.context_tokenizer          = None   
        self.query_text_encoder         = None 
        self.query_text_encoder_linear  = None
        
        self.skiplist = []
        
    def load_model_cpu(self):
        print('begin loading to cpu')
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        print('get config!')
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="query_tokenizer")
        print('init tokenizer')
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False
        print('finished checking instructions')    
        
        self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
        print('query text encoder init')
        self.query_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)
        print('query text linear init')
        
        try:
            self.query_text_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=True))
            self.query_text_encoder_linear.load_state_dict(torch.load(self.local_projection_path, weights_only=True))
        except:
            print(f'Failed to load models checkpoint!!! \n Please check {self.local_encoder_path} or {self.local_projection_path}')

    def load_model_gpu(self):
        self.query_text_encoder_linear.to('cuda')
        self.query_text_encoder.to('cuda')

    def ocdpo_handler(self,**kwargs):
        '''
        The off-critical data path handler
        '''
        blob = kwargs["blob"]
        string_list = deserialize_string_list(blob.tobytes())
        print(f"I recieved kwargs: {string_list}")
        if self.query_text_encoder_linear == None:
            print('==========start loading model cpu==========')
            self.load_model_cpu()
            print('==========start loading model gpu==========')
            self.load_model_gpu()
        print('==========begin step A forward==========')
        encoded_inputs      = self.query_tokenizer(string_list)
        input_ids           = encoded_inputs['input_ids'].to(self.query_text_encoder.device)
        attention_mask      = encoded_inputs['attention_mask'].to(self.query_text_encoder.device)
        
        text_encoder_outputs = self.query_text_encoder(input_ids=input_ids,attention_mask=attention_mask,)
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)
        print('==========finished forward pass==========')
        print(f'text embedding of shape: \t {text_embeddings.shape}')
        print(f'input ids of shape: \t\t {text_embeddings.shape}')
        print(f'hidden sates of shape:\t{text_encoder_hidden_states.shape}')

    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass