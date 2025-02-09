#!/usr/bin/env python3

import numpy as np
import os
import torch
from derecho.cascade.external_client import ServiceClientAPI
import struct
from PIL import Image
import sys
import json


if __name__ == "__main__":
    capi            = ServiceClientAPI()
    key             = "/stepE/1"
    subgroup_type   = "VolatileCascadeStoreWithStringKey"
    subgroup_index  = 0
    shard_index     = 0
    
    
    question_ids = [0, 1]
    dummy_dict = {
        'question_id': question_ids,
        'question': ["test sentence test sentece, this this, 100", "GOJI", "I love puppies"],
        'embedding': [torch.randn(320, 128).tolist() for i in range(len(question_ids))]
    }
    json_string = json.dumps(dummy_dict)
    byte_data = json_string.encode('utf-8')
    
    res = capi.put(key, byte_data, subgroup_type=subgroup_type,
                    subgroup_index=subgroup_index,shard_index=shard_index, message_id=1)