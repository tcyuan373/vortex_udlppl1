#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os, sys
import struct
import torch
from easydict import EasyDict
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger
from transformers import AutoImageProcessor
from PIL import Image
from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
)
from datasets import load_dataset
import time
from serialize_utils import PixelValueBatcher, TextDataBatcher
from torch.utils.data import DataLoader
# import faiss

image_root_dir = "/mydata/EVQA/"
ds_dir = "/mydata/EVQA/EVQA_data/"
STEPA_SHARD_INDICES = [2]
STEPB_SHARD_INDICES = [0, 1]
STEPA_SUBGROUP_INDEX = 0
STEPB_SUBGROUP_INDEX = 0

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

    return pixel_values.numpy().tobytes()


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
    
def add_path_prefix_in_img_path(example, prefix):
        if example["img_path"] != None:
            example["img_path"] = os.path.join(prefix, example["img_path"])
        return example
    
    
def batch_iter(dataset, batch_size):
    """Yield batches sequentially from the dataset."""
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

    
        
        
if __name__ == "__main__":
    tl = TimestampLogger()
    capi = ServiceClientAPI()
    stepa_prefix = "/stepA/"
    stepb_prefix = "/stepB/"
    subgroup_type = "VolatileCascadeStoreWithStringKey"
    
    BS = 1
    num_batches = 1000
    
    # directories and str configs
    image_processor_name = 'openai/clip-vit-large-patch14'
    checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
    
    use_split = "train"
    
    # model configs, tokenziers
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
    
    # TODO: change to actual range at perf test
    ds = load_dataset('parquet', data_files ={  
                                            'train' : ds_dir + 'train-00000-of-00001.parquet',
                                            'test'  : ds_dir + 'test-00000-of-00001-2.parquet',
                                            })[use_split].select(i for i in range(166000, 167000, 1)) 
    # preprocess datasets so that we have 
    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
    ds = ds.map(prepare_inputs)
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=16,
        num_proc=16,
    )
    ds.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "pixel_values", "text_sequence", "question_id", "question"]
    )


    # Create a DataLoader for sequential access with prefetching
    loader = DataLoader(
        ds, 
        batch_size=BS, 
        shuffle=False, 
        num_workers=16,      # Use multiple workers to prefetch batches in parallel
        prefetch_factor=2,   # How many batches each worker preloads (can adjust based on your system)
        pin_memory=True      # Optionally, if you are transferring to GPU later
    )

    # for idx in range(0, len(ds), batch_size):
    #     # TODO: change to the actual at perf test
    #     # idx = torch.randint(0, 99, (1,)).item()
    #     batch = ds[idx : idx + batch_size]
    #     batch_idx = idx // batch_size
    #     # print(f"got batch {batch} with idx being {idx} and {idx+batch_size}")
    #     if batch_idx >= num_batches:    
    #         # print(f"Batch no. {i // batch_size} reached!  Now break")
    #         break
        
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
  
        stepa_serializer = TextDataBatcher()
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = int(qid[uds_idx+1:])
            stepa_serializer.question_ids.append(question_id)
            tl.log(1000, question_id, 0, 0)
        
        stepa_serializer.text_sequence = batch["question"]
        stepa_serializer.input_ids = batch["input_ids"].numpy()
        stepa_serializer.attention_mask = batch["attention_mask"].numpy()
        stepa_serialized_np = stepa_serializer.serialize()
        stepa_key = stepa_prefix + f"_{batch_idx}"
        
        stepa_next_shard_idx = STEPA_SHARD_INDICES[(batch_idx) % len(STEPA_SHARD_INDICES)]
        # tl.log(10000 ,batch_idx ,0 ,0 )
        resA = capi.put_nparray(stepa_key, stepa_serialized_np,subgroup_type=subgroup_type,
                    subgroup_index=STEPA_SUBGROUP_INDEX,shard_index=stepa_next_shard_idx, message_id=1, as_trigger=True, blokcing=False)
        

        stepb_key = stepb_prefix + f"_{batch_idx}"
        serializer = PixelValueBatcher()
        serializer.question_ids = np.asarray(stepa_serializer.question_ids,dtype=np.int64)
        serializer.pixel_values = batch["pixel_values"].numpy()
        serialized_np = serializer.serialize()
        # print(f"With serializer, we got message size of: {sys.getsizeof(serialized_np.tobytes())}")
        stepb_next_shard_idx = STEPB_SHARD_INDICES[(batch_idx) % len(STEPB_SHARD_INDICES)]
        
        
        
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = int(qid[uds_idx+1:])
            # stepa_serializer.question_ids.append(question_id)
            tl.log(10001, question_id, 0, 0)
        resB = capi.put_nparray(stepb_key, serialized_np,subgroup_type=subgroup_type,
                    subgroup_index=STEPB_SUBGROUP_INDEX,shard_index=stepb_next_shard_idx, message_id=1, as_trigger=True, blokcing=False)
        
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = int(qid[uds_idx+1:])
            # stepa_serializer.question_ids.append(question_id)
            tl.log(10020, question_id, 0, 0)
        
        
        time.sleep(0.0002)
        
    tl.flush("client_timestamp.dat")
        # time.sleep(1000)
    # for i in range(10):
    #     key = prefix + f"_{i}"
    #     res = capi.put(key, serialize_string_list(value),subgroup_type=subgroup_type,
    #                 subgroup_index=subgroup_index,shard_index=shard_index, message_id=1)


    # sleep(1000)
  



