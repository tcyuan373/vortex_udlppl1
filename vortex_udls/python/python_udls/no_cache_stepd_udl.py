#!/usr/bin/env python3
import numpy as np
import json
import threading
import torch
import time

import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from TransformerMappingNetwork import MLP, TransformerMappingNetwork
from serialize_utils import (StepCDIntermediateResult, 
                            StepDMessageBatcher,StepBResultBatchManager, 
                            StepAResultBatchManager, PendingStepCDDataBatcher)

STEPD_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
STEPD_NEXT_UDL_SUBGROUP_INDEX = 0
STEPD_NEXT_UDL_PREFIX = "/stepE"

RESULTA_PREFIX = "/no_cache_stepD/resultA_"
RESULTB_PREFIX = "/no_cache_stepD/resultB_"
NUM_TRIES = 5

class StepCDModelWorker:
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.mlp_model = MLP(self.parent.checkpoint_path, self.parent.local_stepc_model_path)
        self.transformer_mapping_model = TransformerMappingNetwork(self.parent.checkpoint_path,
                                                                   self.parent.local_tf_mapping_path, 
                                                                   self.parent.local_tf_mapping_output_path)
    
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.pending_batches = [PendingStepCDDataBatcher(self.max_exe_batch_size) for _ in range(self.parent.num_pending_buffer)]

        
        self.current_batch = -1    # current batch idx that main is executing
        self.next_batch = 0        # next batch idx to add new data
        self.next_to_process = 0  
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = False
        
        self.new_space_available = True
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.main_loop)
        self.thread.start()
    
    def join(self):
        if self.thread is not None:
            self.thread.join()
    
    def signal_stop(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()


    def push_to_pending_batches(self, intermediate_result):
        '''
        Adding the intermediate result one-by-one since they are aggregated one-by-one at the UDL side
        '''
        with self.cv:
            while True:
                if not self.new_space_available:
                    self.cv.wait(timeout=3)
                if not self.running:
                    break
                if self.new_space_available:
                    break
            free_batch = self.next_batch
            space_left = self.pending_batches[free_batch].space_left()
            initial_batch = free_batch
            # Find the idx in the pending_batches to add the data
            while space_left == 0:
                free_batch = (free_batch + 1) % len(self.pending_batches)
                if free_batch == self.current_batch:
                    free_batch = (free_batch + 1) % len(self.pending_batches)
                if free_batch == initial_batch:
                    break
                space_left = self.pending_batches[free_batch].space_left()
            if space_left != 0:
                # add the intermediate result to the pending batch
                self.pending_batches[free_batch].add_data(intermediate_result)
                
                self.next_batch = free_batch
                #  if we complete filled the buffer, cycle to the next
                if self.pending_batches[free_batch].space_left() == 0:
                    self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    if self.next_batch == self.current_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                self.cv.notify()
            else:
                self.new_space_available = False
            

    def main_loop(self):
        batch = None
        while self.running:
            if not batch is None:
                batch.reset()
            with self.cv:
                self.current_batch = -1
                if self.pending_batches[self.next_to_process].num_pending == 0:
                    self.cv.wait(timeout=self.batch_time_us/1000000)
                    
                if self.pending_batches[self.next_to_process].num_pending != 0:
                    self.current_batch = self.next_to_process
                    self.next_to_process = (self.next_to_process + 1) % len(self.pending_batches)
                    batch = self.pending_batches[self.current_batch]
                    
                    if self.current_batch == self.next_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches) 
                    self.new_space_available = True
                    self.cv.notify()
                    
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
                        
            # Execute the batch
            for qid in batch.question_ids:
                self.parent.tl.log(30030, qid, 0, batch.num_pending)
            transformer_mapping_input_features = self.mlp_model.execMLP(batch.vision_second_last_layer_hidden_states[:batch.num_pending,:,:])
            query_embeddings = self.transformer_mapping_model.execTransformerMappingNetwork(
                                batch.input_ids[:batch.num_pending,:], 
                                batch.text_embeddings[:batch.num_pending,:,:], 
                                batch.text_encoder_hidden_states[:batch.num_pending,:,:], 
                                batch.vision_embeddings[:batch.num_pending,:,:],
                                transformer_mapping_input_features)
            query_embeddings = query_embeddings.cpu().detach().numpy()
            
            for qid in batch.question_ids:
                self.parent.tl.log(30031, qid, 0, batch.num_pending)
            
            self.parent.emit_worker.add_to_buffer(batch.question_ids, 
                                                             query_embeddings, 
                                                             batch.np_text_sequence_bytes, 
                                                             batch.num_pending)
            self.pending_batches[self.current_batch].reset()


class StepCDEmitWorker:
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.send_buffer = [StepDMessageBatcher(max_batch_size = self.max_emit_batch_size) for _ in range(self.parent.num_pending_buffer)]
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        # Batch send similar logic as batch exec
        self.current_batch = -1    
        self.next_batch = 0   
        self.next_to_process = 0
        self.running = False
        
        self.sent_batch_counter = 0
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.main_loop)
        self.thread.start()
    
    def join(self):
        if self.thread is not None:
            self.thread.join()
    
    def signal_stop(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()

    def add_to_buffer(self, question_ids, query_embeddings, text_sequence_np, num_pending):
        '''
        pass by object reference to avoid deep-copy
        '''
        question_added = 0
        with self.cv:
            while question_added < num_pending:
                free_batch = self.next_batch
                space_left = self.send_buffer[free_batch].space_left()
                # Find the idx in the pending_batches to add the data
                while space_left == 0:
                    free_batch = (free_batch + 1) % len(self.send_buffer)
                    if free_batch == self.current_batch:
                        free_batch = (free_batch + 1) % len(self.send_buffer)
                    if free_batch >= self.next_batch:
                        break
                    space_left = self.send_buffer[free_batch].space_left()
                if space_left == 0:
                    new_batch = StepDMessageBatcher(max_batch_size = self.max_emit_batch_size)
                    self.send_buffer.append(new_batch)  
                    free_batch = len(self.send_buffer) - 1
                    space_left = self.send_buffer[free_batch].space_left()
                # Add data to send_buffer
                end_idx = self.send_buffer[free_batch].add_results(question_ids, 
                                                                   text_sequence_np,
                                                                   query_embeddings, 
                                                                   question_added)
                question_added = end_idx
                self.next_batch = free_batch
                #  if we complete filled the buffer, cycle to the next
                if self.send_buffer[free_batch].space_left() == 0:
                    self.next_batch = (self.next_batch + 1) % len(self.send_buffer)
                    if self.next_batch == self.current_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.send_buffer)
            self.cv.notify()
        
    
    def main_loop(self):
        batch = None
        while self.running:
            if not batch is None:
                batch.reset()
            with self.cv:
                self.current_batch = -1
                if self.send_buffer[self.next_to_process].num_queries == 0:
                    self.cv.wait(timeout=self.parent.batch_time_us/1000000)
                    
                if self.send_buffer[self.next_to_process].num_queries != 0:
                    self.current_batch = self.next_to_process
                    self.next_to_process = (self.next_to_process + 1) % len(self.send_buffer)
                    batch = self.send_buffer[self.current_batch]
                    
                    if self.current_batch == self.next_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.send_buffer) 
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            
            # serialize the batch
            batch_np = batch.serialize()
            # send to a evenly chosen shard
            weighted_pos = self.parent.weighted_indices[self.sent_batch_counter % len(self.parent.weighted_indices)]
            shard_idx = self.parent.stepd_next_udl_shards[weighted_pos]
            new_key = STEPD_NEXT_UDL_PREFIX + "/stepD_{self.sent_batch_counter}"
            
            for qid in batch.question_ids:
                self.parent.tl.log(30100, qid, 0, batch.num_queries)
                
            self.parent.capi.put_nparray(new_key, batch_np, 
                                subgroup_type=STEPD_NEXT_UDL_SUBGROUP_TYPE,
                                subgroup_index=STEPD_NEXT_UDL_SUBGROUP_INDEX, 
                                shard_index=shard_idx, 
                                message_id=1, as_trigger=True, blocking=False)
            self.sent_batch_counter += 1
            self.send_buffer[self.current_batch].reset()
            
            
        


class StepCDUDL(UserDefinedLogic):
    '''
    StepCD performs aggregation, then MLP and TransformerMappingNetwork on the aggregated results
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepCDUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        
        self.checkpoint_path = self.conf["checkpoint_path"]
        self.local_stepc_model_path = self.conf["local_stepc_model_path"]
        self.local_tf_mapping_path = self.conf["local_tf_mapping_path"]
        self.local_tf_mapping_output_path = self.conf["local_tf_mapping_output_path"]

        self.max_exe_batch_size = self.conf["max_exe_batch_size"]
        self.batch_time_us = self.conf["batch_time_us"]
        self.max_emit_batch_size = self.conf["max_emit_batch_size"]
        
        self.stepd_next_udl_shards = self.conf.get("stepd_next_udl_shards", [2])
        # Initial number of pending batches smaller, because they are allocated on GPU, 
        # if the max_exec_batch_size is 16, it takes 20 * 18MB memory on GPU
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 20)
        self.weighted_indices = self.conf.get("weighted_indices", [0])
        # Keep track of collected intermediate results: {query_id0: StepCDIntermediateResult, query_id2:{} ...}
        self.collected_intermediate_results = {}
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
    
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = StepCDModelWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = StepCDEmitWorker(self, 2)
            self.emit_worker.start()

    def append_result_to_collector(self, key, blob):
        
        step_A_idx = key.find("resultA") 
        step_B_idx = key.find("resultB")
        
        collect_all = False
        stepCD_intermediate_result = StepCDIntermediateResult()
        # if step_A_idx != -1:
        #     stepa_serializer = StepAResultBatchManager()
            
        #     qid = int(key[key.find("resultA_")+8:])
        #     self.tl.log(30000, qid, 1, 0)
            
        #     stepB_result_key = RESULTB_PREFIX + str(qid)
        #     for i in range(NUM_TRIES):
        #         # Get stepB result
        #         res = self.capi.get(stepB_result_key)
        #         if res:
        #             odict = res.get_result()
        #             blob_b = odict["value"]
        #             if len(blob_b) == 0:
        #                 continue
                    
        #             stepa_serializer.deserialize(blob)
        #             stepCD_intermediate_result._question_id = qid
        #             stepCD_intermediate_result._np_text_sequence_bytes = np.copy(stepa_serializer.np_text_sequence_bytes[0])
        #             stepCD_intermediate_result._input_ids = torch.Tensor(stepa_serializer.input_ids[0])
        #             stepCD_intermediate_result._text_embeddings = torch.Tensor(stepa_serializer.text_embeds[0])
        #             stepCD_intermediate_result._text_encoder_hidden_states = torch.Tensor(stepa_serializer.text_encoder_hidden_states[0])
                    
        #             # Deserialize the stepB result
        #             stepb_batcher = StepBResultBatchManager()
        #             arr_b = np.frombuffer(blob_b, dtype=np.uint8)
        #             stepb_batcher.deserialize(arr_b)
        #             stepCD_intermediate_result._vision_embeddings = torch.Tensor(stepb_batcher.vision_embedding[0])
        #             stepCD_intermediate_result._vision_second_last_layer_hidden_states = torch.Tensor(stepb_batcher.vision_second_last_layer_hidden_states[0])
        #             collect_all =  True
        #             break
        #         else:
        #             time.sleep(0.000001)
                
        
            
        if step_B_idx != -1:
            stepb_batcher = StepBResultBatchManager()
            qid = int(key[key.find("resultB_")+8:])
            self.tl.log(30010, qid, 2, 0)
            stepA_result_key = RESULTA_PREFIX + str(qid)
            for i in range(NUM_TRIES):
                # Get stepA result
                res = self.capi.get(stepA_result_key)
                if res:
                    
                    odict = res.get_result()
                    blob_a = odict["value"]
                    if len(blob_a) == 0:
                        continue
                    
                    stepb_batcher.deserialize(blob)
                    
                    stepCD_intermediate_result._question_id = qid
                    stepCD_intermediate_result._vision_embeddings = torch.Tensor(stepb_batcher.vision_embedding[0])
                    stepCD_intermediate_result._vision_second_last_layer_hidden_states = torch.Tensor(stepb_batcher.vision_second_last_layer_hidden_states[0])
                    
                    stepa_batcher = StepAResultBatchManager()
                    arr_a = np.frombuffer(blob_a, dtype=np.uint8)
                    stepa_batcher.deserialize(arr_a)
                    stepCD_intermediate_result._input_ids = torch.Tensor(stepa_batcher.input_ids[0])
                    stepCD_intermediate_result._text_embeddings = torch.Tensor(stepa_batcher.text_embeds[0])
                    stepCD_intermediate_result._text_encoder_hidden_states = torch.Tensor(stepa_batcher.text_encoder_hidden_states[0])
                    stepCD_intermediate_result._np_text_sequence_bytes = np.copy(stepa_batcher.np_text_sequence_bytes[0])
                    
                    collect_all = True
                    break
                else:
                    time.sleep(0.000001)
        
        if collect_all:
            self.tl.log(30011, qid, 3, 0)
            self.model_worker.push_to_pending_batches(stepCD_intermediate_result)
        
        
    def ocdpo_handler(self, **kwargs):
        # preparing for input : 
            # (input_ids                              # step A      B * seq_len
            # text_embeddings                         # step A      B * 32 * 128
            # text_encoder_hidden_states              # step A      B * 32 * 768
            # vision_embeddings                       # step B      B * 32 * 128
            # vision_second_last_layer_hidden_states)     # step B  B * 256 * 1024
        if not self.model_worker:
            self.start_threads()
            
        key = kwargs["key"]
        blob = kwargs["blob"]
        
        self.append_result_to_collector(key, blob)
        
        

        


        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"StepCDUDL destructor")
        pass