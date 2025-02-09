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
from flmr import FLMRConfig, FLMRVisionModel, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel
from transformers import AutoImageProcessor
from step_C_modeling_mlp import StepC




class FLMRMultiLayerPerceptron(nn.Module):
    """
    A simple multi-layer perceptron with an activation function. This can be used as the mapping network in the FLMR model.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(FLMRMultiLayerPerceptron, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        
        
class StepBUDL(UserDefinedLogic):
    '''
    ConsolePrinter is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepBUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        self.checkpoint_path            = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_encoder_path         = 'perf_data/pipeline1/models/models_step_B_vision_encoder.pt'
        self.local_projection_path      = 'perf_data/pipeline1/models/models_step_B_vision_projection.pt'
        self.image_processor            = None
        self.flmr_config                = None

        self.query_vision_encoder       = None
        self.query_vision_projection    = None
        self.skiplist = []
        self.device = None
        self.stepc = StepC()
        
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.query_vision_encoder = FLMRVisionModel(self.flmr_config.vision_config)
        self.query_vision_projection = FLMRMultiLayerPerceptron(
                (
                    self.flmr_config.vision_config.hidden_size,
                    (self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length) // 2,
                    self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length,
                )
            )
        self.query_vision_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=False))
        self.query_vision_projection.load_state_dict(torch.load(self.local_projection_path, weights_only=False))
    
    def load_model_gpu(self):
        self.query_vision_projection.cuda()
        self.query_vision_encoder.cuda()
        self.device = self.query_vision_encoder.device
        
    def ocdpo_handler(self,**kwargs):
        '''
        The off-critical data path handler
        '''
        blob = kwargs["blob"]
        # list_of_images = deserialize_string_list(blob.tobytes())
        # should be a 5D tensor of shape B * 1 * n_channel(3) * H * W
        reconstructed_np_array = np.frombuffer(blob, dtype=np.float32).reshape(-1, 1, 3, 224, 224)
        input_tensor = torch.Tensor(reconstructed_np_array)
        print(f"got input tensor of shape: {reconstructed_np_array.shape}")
        if self.query_vision_projection == None:
            print('==========start loading model cpu==========')
            self.load_model_cpu()
            self.stepc.load_model_cpu()
            print('==========start loading model gpu==========')
            self.load_model_gpu()
            self.stepc.load_model_gpu()
            
        
        batch_size = input_tensor.shape[0]
        # Forward the vision encoder
        print('==========begin step B forward==========')
        input_tensor = input_tensor.to(self.device)
        if len(input_tensor.shape) == 5:
            # Multiple ROIs are provided
            # merge the first two dimensions
            input_tensor = input_tensor.reshape(
                -1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]
            )
        vision_encoder_outputs = self.query_vision_encoder(input_tensor, output_hidden_states=True)
        vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]
        vision_embeddings = self.query_vision_projection(vision_embeddings)
        vision_embeddings = vision_embeddings.view(batch_size, -1, self.flmr_config.dim)
    
        vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[-2][:, 1:]
        
        print('==========Step B Finished ==========')
        print(f'vision_embeddings has shape {vision_embeddings.shape} | vision hidden states has shape{vision_second_last_layer_hidden_states.shape}')
        
        stepcOUTPUT = self.stepc.stepC_output(vision_second_last_layer_hidden_states)
        print('==========Step C Finished==========')
        print(f'Transformer_mapping_input_features has shape: {stepcOUTPUT.shape}')
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass
    