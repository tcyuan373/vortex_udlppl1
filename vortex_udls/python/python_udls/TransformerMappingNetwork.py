import torch, os
import torch
from torch import Tensor, nn
from flmr import FLMRConfig
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

class MLP:
    '''
    stepC model definition
    '''
    def __init__(self, checkpoint_path, local_stepc_model_path):
        self.checkpoint_path = checkpoint_path
        self.local_stepc_model_path = local_stepc_model_path
        self.flmr_config = None
        self.transformer_mapping_input_linear = None
        if not os.path.exists(self.local_stepc_model_path):
            print(f'local directory not found, initing from full model...')
            self.acquire_and_save_module()
            

    def acquire_and_save_module(self):
        '''
        This function should not be called usually, only when the local model is not found
        '''
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
            self.checkpoint_path, 
            text_config=self.flmr_config.text_config, 
            subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
            self.checkpoint_path, 
            text_config=self.flmr_config.text_config, 
            subfolder="context_tokenizer"
        )
        full_model = FLMRModelForRetrieval.from_pretrained(
            self.checkpoint_path,
            query_tokenizer=self.query_tokenizer,
            context_tokenizer=self.context_tokenizer,
        )
        self.transformer_mapping_input_linear = full_model.transformer_mapping_input_linear
        # torch.save(self.transformer_mapping_input_linear.state_dict(), self.local_model_path)
        del full_model
            

    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        transformer_mapping_config_base = self.flmr_config.transformer_mapping_config_base
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        transformer_mapping_config.num_hidden_layers = self.flmr_config.transformer_mapping_num_hidden_layers
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        print(f'found local model for step C, now loading...')
        self.transformer_mapping_input_linear = nn.Linear(
            self.flmr_config.vision_config.hidden_size, transformer_mapping_config.hidden_size
        )
        self.transformer_mapping_input_linear.load_state_dict(torch.load(self.local_stepc_model_path, weights_only=True))
        
    def load_model_gpu(self):
        self.transformer_mapping_input_linear.cuda()
        

    def execMLP(self, vision_second_last_layer_hidden_states):
        if self.transformer_mapping_input_linear == None:
            self.load_model_cpu()
            self.load_model_gpu()
        transformer_mapping_input_features = self.transformer_mapping_input_linear(
            vision_second_last_layer_hidden_states
        )
        
        return transformer_mapping_input_features
    

class TransformerMappingNetwork:
    '''
    StepD 
    '''
    def __init__(self, checkpoint_path,
                 local_tf_mapping_path, local_tf_mapping_output_path):
        # modeling configs
        self.flmr_config = None
        self.skiplist = []
        self.query_tokenizer = None
       
        self.transformer_mapping_cross_attention_length = 32
        self.vision_encoder_embedding_size = 1024
        self.late_interaction_embedding_size = 128
        self.checkpoint_path = checkpoint_path
        self.transformer_mapping_config_base = 'bert-base-uncased'
        self.local_tf_mapping_path = local_tf_mapping_path
        self.local_tf_mapping_output_path = local_tf_mapping_output_path
        self.transformer_mapping_network = None
        self.transformer_mapping_output_linear = None
        
        self.mask_instruction = False
        
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        transformer_mapping_config = BertConfig.from_pretrained(self.transformer_mapping_config_base)
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        transformer_mapping_config.num_hidden_layers = 1

        self.transformer_mapping_network = BertEncoder(transformer_mapping_config)
        self.transformer_mapping_network.load_state_dict(torch.load(self.local_tf_mapping_path, weights_only=True))
        self.transformer_mapping_output_linear = nn.Linear(
            transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
        )
        self.transformer_mapping_output_linear.load_state_dict(torch.load(self.local_tf_mapping_output_path, weights_only=True))
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                    self.checkpoint_path, 
                    text_config=self.flmr_config.text_config, 
                    subfolder="query_tokenizer")
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False
        
    def mask(self, input_ids, skiplist):
        return [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
    
    
    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return self.mask(input_ids, skiplist)

        # find the position of end of instruction in input_ids
        # mask the tokens before the position
        sep_id = self.instruction_token_id
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        # if any of the positions is lower than 1, set to 1
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
        mask = [
            [
                (x not in skiplist) and (x != 0) and (index > sep_positions[seq_index] or index < 2)
                for index, x in enumerate(d)
            ]
            for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        return mask
        
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_attention_mask.dtype).min

        return encoder_extended_attention_mask
    
    
    def load_model_gpu(self):
        self.transformer_mapping_network.cuda()
        self.transformer_mapping_output_linear.cuda()
        
    
        
    def execTransformerMappingNetwork(self,
                       input_ids,
                       text_embeddings,
                       text_encoder_hidden_states,
                       vision_embeddings,
                       transformer_mapping_input_features,
                       ):
        
        if self.transformer_mapping_network == None:
            # print('==========start loading model cpu==========')
            self.load_model_cpu()
            
            # print('==========start loading model gpu==========')
            self.load_model_gpu()
            
        mask = torch.as_tensor(self.query_mask(input_ids, skiplist=self.skiplist)).unsqueeze(2).float().cuda()
        
        text_embeddings = text_embeddings.to(mask.device) * mask
        encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
        
        if text_encoder_hidden_states.shape[1] > self.transformer_mapping_cross_attention_length:
            text_encoder_hidden_states = text_encoder_hidden_states[:, :self.transformer_mapping_cross_attention_length]
            encoder_mask = encoder_mask[:, :self.transformer_mapping_cross_attention_length]
        # Obtain cross attention mask
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
        # Pass through the transformer mapping
        
        # ENCODER hidden states: Encoder_bsize, Encoder_seqLen, _
        # ENCODER attention mask: ones_like(encoder_hidden_states)
        
        # Note: transformer_mapping_input_features, encoder_hidden_states, encoder_attention_mask are already on GPU 
        transformer_mapping_outputs = self.transformer_mapping_network(
            transformer_mapping_input_features,
            encoder_hidden_states=text_encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        transformer_mapping_output_features = transformer_mapping_outputs.last_hidden_state
        # Convert the dimension to FLMR dim
        transformer_mapping_output_features = self.transformer_mapping_output_linear(
            transformer_mapping_output_features
        )
        # Merge with the vision embeddings
        
        vision_embeddings = torch.cat([vision_embeddings.to(mask.device), transformer_mapping_output_features], dim=1)
        
        Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        query_embeddings = torch.nn.functional.normalize(Q, p=2, dim=2)
        return query_embeddings


# if __name__ == "__main__": # Bsize, vision_hidden_size[-2], vision_hidden_size[-1]
#     stepc = StepC()
#     stepc.load_model_cuda()
#     bsize = 16
#     dummy_hidden_states = torch.randn(bsize, 256, 1024).cuda()
#     output = stepc.stepC_output(dummy_hidden_states)
#     output.cpu()
#     print(f"transformer mapping input feature shape is: {output.shape}")