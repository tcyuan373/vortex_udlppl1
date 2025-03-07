from flmr import FLMRConfig, FLMRVisionModel
from transformers import AutoImageProcessor
import torch
from torch import nn

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
        

class VisionEncoder:
    def __init__(self, checkpoint_path, local_encoder_path, local_projection_path):
        self.checkpoint_path            = checkpoint_path
        self.local_encoder_path         = local_encoder_path
        self.local_projection_path      = local_projection_path
        self.image_processor            = None
        self.flmr_config                = None

        self.query_vision_encoder       = None
        self.query_vision_projection    = None
        self.skiplist = []
        self.device = None


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
        
    def execVisionEncoder(self, input_tensor, batch_size):
        if len(input_tensor.shape) == 5:
            # Multiple ROIs are provided
            # merge the first two dimensions
            input_tensor = input_tensor.reshape(
                -1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]
            )
        if self.query_vision_projection == None:
            self.load_model_cpu()
            self.load_model_gpu()
        vision_encoder_outputs = self.query_vision_encoder(input_tensor, output_hidden_states=True)
        vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]
        vision_embeddings = self.query_vision_projection(vision_embeddings)
        vision_embeddings = vision_embeddings.view(batch_size, -1, self.flmr_config.dim)
        vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[-2][:, 1:]
        
        return vision_embeddings, vision_second_last_layer_hidden_states
        