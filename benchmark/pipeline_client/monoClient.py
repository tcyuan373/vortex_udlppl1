
import numpy as np
import os
import torch
from derecho.cascade.external_client import ServiceClientAPI
import struct
from transformers import AutoImageProcessor
from PIL import Image
import sys, json


def serialize_string_list(string_list):
    """Serialize a list of strings into a custom binary format."""
    encoded_strings = [s.encode('utf-8') for s in string_list]
    offsets = []
    current_offset = 0

    # Compute the offsets for each string
    for s in encoded_strings:
        offsets.append(current_offset)
        current_offset += len(s)

    # Pack the number of elements
    header = struct.pack("I", len(string_list))  # 4 bytes for length
    offset_section = struct.pack(f"{len(offsets)}I", *offsets)  # 4 bytes per offset

    # Concatenate everything into a byte stream
    serialized_data = header + offset_section + b''.join(encoded_strings)
    return serialized_data


def process_img_2_nparray(img_root, image_processor):
    img_paths = [os.path.join(img_root, item) for item in os.listdir(img_root)]
    list_of_images = []
    pixel_values = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        list_of_images.append(image)
   
    for img in list_of_images:
        encoded = image_processor(img, return_tensors="pt")
        pixel_values.append(encoded.pixel_values)
    pixel_values = torch.stack(pixel_values, dim=0)

    return pixel_values.tolist()


def prepare_inputs(sample):
    sample = EasyDict(sample)

    module = EasyDict(
        {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
    )

    instruction = sample.instruction.strip()
    if instruction[-1] != ":":
        instruction = instruction + ":"
    instruction = instruction.replace(":", flmr_config.mask_instruction_token)
    #random_instruction = random.choice(instructions)
    text_sequence = " ".join(
        [instruction]
        + [module.separation_tokens.start]
        + [sample.question]
        + [module.separation_tokens.end]
    )

    sample["text_sequence"] = text_sequence

    return sample


def tokenize_inputs(examples, query_tokenizer, image_processor):
    encoding = query_tokenizer(examples["text_sequence"])
    examples["input_ids"] = encoding["input_ids"]
    examples["attention_mask"] = encoding["attention_mask"]

    pixel_values = []
    for img_path in examples["img_path"]:

        if img_path is None:
            image = Image.new("RGB", (336, 336), color='black')
        else:
            image = Image.open(img_path).convert("RGB")
        
        encoded = image_processor(image, return_tensors="pt")
        pixel_values.append(encoded.pixel_values)

    pixel_values = torch.stack(pixel_values, dim=0)
    examples["pixel_values"] = pixel_values
    return examples



if __name__ == "__main__":
    # prepare inputs
    # tokenize inputs
    # pass to UDL
    capi            = ServiceClientAPI()
    key             = "/Mono/1"
    subgroup_type   = "VolatileCascadeStoreWithStringKey"
    subgroup_index  = 0
    shard_index     = 0
    
    data = {}
    img_root        = './perf_data/pipeline1/data/images'
    image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14', use_fast=True)
    processed_img_nparray = process_img_2_nparray(img_root, image_processor)
    
    texts = ["Is this plant poisonous?"]
    
    data["input_texts"] = texts
    data["pixel_values"] = processed_img_nparray
    data["question_id"] = ["EVQA_0"]
    data["question"] = ["Is this plant poisonous?"]
    
    
    json_str = json.dumps(data)
    byte_data = json_str.encode('utf-8')
    
    res = capi.put(key, byte_data, subgroup_type=subgroup_type,
                subgroup_index=subgroup_index,shard_index=shard_index, message_id=1, trigger=True)