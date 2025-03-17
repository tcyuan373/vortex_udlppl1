#!/usr/bin/env python3
import numpy as np
import os, time
import struct
import torch
import json
from easydict import EasyDict

from derecho.cascade.external_client import TimestampLogger
from transformers import AutoImageProcessor
from PIL import Image
from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
)
from datasets import load_dataset
from serialize_utils import MonoDataBatcher
from torch.utils.data import DataLoader
    
    
def get_images(example):
    images = []
    for img_path in example["img_path"]:
        if img_path is None:
            image = Image.new("RGB", (336, 336), color='black')
        else:
            image = Image.open(img_path).convert("RGB")
        images.append(image)
    return images


def add_path_prefix_in_img_path(example, prefix):
    if example["img_path"] != None:
        example["img_path"] = os.path.join(prefix, example["img_path"])
    return example



if __name__ == "__main__":
    # prepare inputs
    # tokenize inputs
    # pass to UDL
    tl              = TimestampLogger()
    prefix          = "/Mono/"
    subgroup_type   = "VolatileCascadeStoreWithStringKey"
    subgroup_index  = 0
    BS              = 1
    num_batches     = 1000
    
    checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
    image_processor_name = 'openai/clip-vit-large-patch14'
    ds_dir = "/mnt/nvme0/vortex_pipeline1/EVQA_data"
    image_root_dir = "/mnt/nvme0/vortex_pipeline1"
    use_split = "train"
    # model configs, tokenziers
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
    
    ds = load_dataset('parquet', data_files ={  
                                            'train' : ds_dir + '/train-00000-of-00001.parquet',
                                            'test'  : ds_dir + '/test-00000-of-00001-2.parquet',
                                            })[use_split].select([i for i in range(166000, 167000, 1)])
    # preprocess datasets so that we have 
    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
    ds = ds.map(prepare_text_sequence)
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

    
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        batcher = MonoDataBatcher()
        # print(f"Got qid list : {batch['question_id']}")
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = qid[uds_idx+1:]
            batcher.question_ids.append(question_id)
            tl.log(1000, int(question_id), 0, 0)
        batcher.attention_mask = batch["attention_mask"].numpy()
        batcher.input_ids = batch["input_ids"].numpy()
        batcher.text_sequence = batch["question"] 
        batcher.pixel_values = batch["pixel_values"].numpy()
        serialized_np = batcher.serialize()
        
        mono_shard_id = MONO_SHARD_IDS[(batch_idx) % len(MONO_SHARD_IDS)]
        
        res = capi.put_nparray(prefix + f"_{batch_idx}", serialized_np, subgroup_type=subgroup_type,
                    subgroup_index=subgroup_index,shard_index=mono_shard_id, message_id=batch_idx, as_trigger=True, blokcing=False)

        time.sleep(0.05)
        
    tl.flush("mono_client_timestamp.dat")