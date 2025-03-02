#!/usr/bin/env python3
import numpy as np
import json
import queue
import threading
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from serialize_utils import TextDataBatcher, StepAMessageDataBatcher, PendingTextDataBatcher
from TextEncoder import TextEncoder

STEPA_NEXT_UDL_SHARD_INDEX = 2
STEPA_WORKER_INITIAL_PENDING_BATCHES = 10


class StepAModelWorker(threading.Thread):
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self, max_batch_size = 10, batch_time_us=10000):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.pending_batches = [PendingTextDataBatcher(max_batch_size) for _ in range(STEPA_WORKER_INITIAL_PENDING_BATCHES)]
        
        self.current_batch = -1    # current batch idx that main is executing
        self.next_batch = 0        # next batch idx to add new data
        self.next_to_process = 0  
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = True
        self.max_batch_size = max_batch_size
        self.batch_time_us = batch_time_us

    def push_to_pending_batches(self, text_data_batcher):
        num_questions = len(text_data_batcher.question_ids)
        question_added = 0
        with self.cv:
            while question_added < num_questions:
                free_batch = self.next_batch
                space_left = self.pending_batches[free_batch].space_left()
                # Find the idx in the pending_batches to add the data
                while space_left == 0:
                    free_batch = (free_batch + 1) % len(self.pending_batches)
                    if free_batch == self.current_batch:
                        free_batch = (free_batch + 1) % len(self.pending_batches)
                    if free_batch >= self.next_batch:
                        break
                    space_left = self.pending_batches[free_batch].space_left()
                if space_left == 0:
                    # Need to create new batch, if all the pending_batches are full
                    new_batch = PendingTextDataBatcher(self.max_batch_size)
                    self.pending_batches.append(new_batch)  
                    free_batch = len(self.pending_batches) - 1
                    space_left = self.pending_batches[free_batch].space_left()
                
                # add as many questions as possible to the pending batch
                self.next_batch = free_batch
                question_start_idx = question_added
                end_idx = self.pending_batches[free_batch].add_data(text_data_batcher, question_start_idx)
                question_added = end_idx
                #  if we complete filled the buffer, cycle to the next
                if self.pending_batches[free_batch].space_left() == 0:
                    self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    if self.next_batch == self.current_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                        
            self.cv.notify()
            print("added to queue")
            

    def run(self):
        batch = None
        while self.running:
            if not batch is None:
                batch.reset()
            with self.cv:
                self.current_batch = -1
                if self.pending_batches[self.next_to_process].num_pending == 0:
                    self.cv.wait(timeout=self.batch_time_us/1000000)
                    
                if not self.pending_batches[self.next_to_process].num_pending == 0:
                    self.current_batch = self.next_to_process
                    self.next_to_process = (self.next_to_process + 1) % len(self.pending_batches)
                    batch = self.pending_batches[self.current_batch]
                    
                    if self.current_batch == self.next_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches) 
                    print("found something to process")
            if not self.running:
                break
            if self.current_batch == -1:
                continue
            print("about to execute")
            # Execute the batch
            text_embeddings, text_encoder_hidden_states = self.text_encoder.execTextEncoder(batch.input_ids[:batch.num_pending], batch.attention_mask[:batch.num_pending])
            
            # push to the batching thread
            # stepa_serializer = StepAMessageDataBatcher()
            # capi.put....         

            batch.reset()


class StepAEmitWorker:
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self):
        self.buffer = [] # TODO: it is a buffer

    def add_to_buffer(self):
        pass
    
    def Emit_to_next_UDL(self):
        pass


class StepAUDL(UserDefinedLogic):
    '''
    StepAUDL is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepAUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        # print(f"StepAUDL constructor received json configuration: {self.conf}")
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        
        # Create and start the worker threads.
        self.model_worker = StepAModelWorker(
            batch_time_us=self.conf.get("batch_time_us", 10000)
        )
        # self.emit_worker = StepAEmitWorker(
        #     self.capi, batch_time_us=self.conf.get("batch_time_us", 10000)
        # )
        self.model_worker.start()
        # self.emit_worker.start()
        
        
    
        
    
    
    



    def ocdpo_handler(self,**kwargs):
        '''
        The off-critical data path handler
        '''
        key = kwargs['key']
        blob = kwargs["blob"]
        
        new_batcher = TextDataBatcher()
        new_batcher.deserialize(blob)
        self.model_worker.push_to_pending_batches(new_batcher)


        return
        # blob_bytes = blob.tobytes()
        # res_json_str = blob_bytes.decode('utf-8')
        # encoded_inputs = json.loads(res_json_str)
        input_ids_np = np.copy(encoded_inputs['input_ids'])   # TODO: take away the copy
        attn_msk_np = np.copy(encoded_inputs['attention_mask'])
        key_id = key[int(key.find('_'))+1:]
        batch_id = int(key_id)
        self.tl.log(20041, batch_id, 0, 0)
        # print('===========Step A start loading model==========')
        if self.query_text_encoder_linear == None:
            self.load_model_cpu()
            self.load_model_gpu()
        # encoded_inputs      = self.query_tokenizer(string_list)
        input_ids           = torch.LongTensor(input_ids_np).to(self.device)
        attention_mask      = torch.Tensor(attn_msk_np).to(self.device)
        # print(f"STEP A Got input ids of shape: {input_ids.shape} | attn_mask of shape: {attention_mask.shape}")
        # text_encoder_outputs = self.query_text_encoder(input_ids=input_ids,attention_mask=attention_mask,)
        # text_encoder_hidden_states = text_encoder_outputs[0]
        # text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)
        # print('==========Step A finished forward pass==========')
        # print(f'text embedding of shape: \t {text_embeddings.shape}')
        # print(f'input ids of shape: \t\t {text_embeddings.shape}')
        # print(f'hidden sates of shape:\t{text_encoder_hidden_states.shape}')
        # result = {}
        # result['queries'] = encoded_inputs["text_sequence"]
        # result['question_id'] = encoded_inputs["question_ids"]
        # result['input_ids'] = input_ids.tolist()
        # result['text_embeddings'] = text_embeddings.tolist()
        # result['text_encoder_hidden_states'] = text_encoder_hidden_states.tolist()
        # res_json_str = json.dumps(result)
        # res_json_byte = res_json_str.encode('utf-8')
        
        stepa_serializer = StepAMessageDataBatcher()
        stepa_serializer.queries = encoded_inputs["text_sequence"]
        stepa_serializer.question_ids = encoded_inputs["question_ids"]
        stepa_serializer.input_ids = input_ids.cpu().detach().numpy()
        stepa_serializer.text_embeds = text_embeddings.cpu().detach().numpy()
        stepa_serializer.text_encoder_hidden_states = text_encoder_hidden_states.cpu().detach().numpy()
        
        stepa_serialized_np = stepa_serializer.serialize()
        subgroup_type = "VolatileCascadeStoreWithStringKey"
        subgroup_index = 0
        prefix = "/stepD/stepA_"
        
        # indices = [i for i, char in enumerate(key) if char == "/"]
        # key_id = key[int(indices[-1]):]
        
        new_key =  prefix + key_id
        res = self.capi.put(new_key, stepa_serialized_np.tobytes(), subgroup_type=subgroup_type,
                subgroup_index=subgroup_index,shard_index=STEPA_NEXT_UDL_SHARD_INDEX, message_id=1)
        self.tl.log(20050, batch_id, 0, 0)
        if batch_id==49:
            self.tl.flush(f"node{self.my_id}_STEPA_udls_timestamp.dat")
            print("STEPA TL flushed!!!")
        
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"StepAUDL destructor")
        pass