from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel
import torch
from torch import Tensor, nn


class TextEncoder:
    def __init__(self):
        self.checkpoint_path            = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_encoder_path         = '/mnt/nvme0/yy354/models/models_pipeline1/models_step_A_query_text_encoder.pt'
        self.local_projection_path      = '/mnt/nvme0/yy354/models/models_pipeline1/models_step_A_query_text_linear.pt'
        self.flmr_config                = None
        self.query_tokenizer            = None
        self.context_tokenizer          = None   
        self.query_text_encoder         = None 
        self.query_text_encoder_linear  = None
        self.device                     = 'cpu'
        self.skiplist                   = []
        
    
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
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
        
        self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
        self.query_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)
        
        try:
            self.query_text_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=True))
            self.query_text_encoder_linear.load_state_dict(torch.load(self.local_projection_path, weights_only=True))
        except:
            print(f'Failed to load models checkpoint!!! \n Please check {self.local_encoder_path} or {self.local_projection_path}')


    def load_model_gpu(self):
        self.device = 'cuda'
        self.query_text_encoder_linear.to(self.device)
        self.query_text_encoder.to(self.device)
        

    def execTextEncoder(self, input_ids, attention_mask):
        '''
        Execute batch of input_ids and attention_mask
        '''
        if self.query_text_encoder_linear == None:
            self.load_model_cpu()
            self.load_model_gpu()
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        print(f"input_id shape: {input_ids.shape} | attention_mask shape: {attention_mask.shape}")
        text_encoder_outputs = self.query_text_encoder(input_ids=input_ids,attention_mask=attention_mask,)
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)
        print('==========Step A finished forward pass==========')
        print(f'text embedding of shape: \t {text_embeddings.shape}')
        print(f'input ids of shape: \t\t {text_embeddings.shape}')
        print(f'hidden sates of shape:\t{text_encoder_hidden_states.shape}')
        return text_embeddings, text_encoder_hidden_states