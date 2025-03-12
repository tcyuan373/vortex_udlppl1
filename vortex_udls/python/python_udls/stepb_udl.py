#!/usr/bin/env python3
import json
import numpy as np
import threading
import torch
import time

from VisionEncoder import VisionEncoder

import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from VisionEncoder import VisionEncoder
from serialize_utils import PixelValueBatcher, PendingVisionDataBatcher, StepBResultBatchManager

STEPB_NEXT_UDL_PREFIX = "/stepD/resultB_"
STEPB_WORKER_INITIAL_PENDING_BATCHES = 3
STEPB_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
STEPB_NEXT_UDL_SUBGROUP_INDEX = 0


class StepBModelWorker:
    '''
    This is a batcher for StepB execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.vision_encoder = VisionEncoder(self.parent.checkpoint_path, self.parent.local_encoder_path, self.parent.local_projection_path)
        self.pending_batches = [PendingVisionDataBatcher(self.max_exe_batch_size) for _ in range(STEPB_WORKER_INITIAL_PENDING_BATCHES)]
        
        self.current_batch = -1    # current batch idx that main is executing
        self.next_batch = 0        # next batch idx to add new data
        self.next_to_process = 0  
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = False
    
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


    def push_to_pending_batches(self, vision_data_batcher):
        for qid in vision_data_batcher.question_ids:
            self.parent.tl.log(20000, qid, 0, self.parent.my_id)
        num_questions = vision_data_batcher.question_ids.shape[0]
        question_added = 0
        while question_added < num_questions:
            with self.cv:
                wait_result = self.cv.wait_for(
                    lambda: any(batch.space_left() > 0
                            for i, batch in enumerate(self.pending_batches)
                            if i != self.current_batch),
                    timeout=self.batch_time_us/1000000
                )
                
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
                if space_left == 0:
                    continue
                    # self.cv.wait(timeout=self.batch_time_us/1000000) # yield to allow processing thread to run
                    # new_batch = PendingVisionDataBatcher(self.max_exe_batch_size)
                    # self.pending_batches.append(new_batch)  
                    # free_batch = len(self.pending_batches) - 1
                    # space_left = self.pending_batches[free_batch].space_left()
                self.next_batch = free_batch
                question_start_idx = question_added
                end_idx = self.pending_batches[free_batch].add_data(vision_data_batcher, question_start_idx)
                question_added = end_idx
                if self.pending_batches[free_batch].space_left() == 0:
                    self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    if self.next_batch == self.current_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                self.cv.notify()
            # # Yield control to allow other threads to run.
            # time.sleep(self.batch_time_us / 2000000)
            
        for qid in vision_data_batcher.question_ids:
            self.parent.tl.log(20050, qid, 0, 0)

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
                        
                    self.pending_batches[self.current_batch] = PendingVisionDataBatcher(self.max_exe_batch_size)
                    self.cv.notify_all()
                    
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(20020, qid, self.next_batch, batch.num_pending)
            # Execute the batch
            # TODO: use direct memory sharing via pointer instead of copying to the host
            input_tensor = torch.as_tensor(batch.pixel_values[:batch.num_pending,:,:,:,:], dtype=torch.long, device="cuda") 
            vision_embeddings, vision_second_last_layer_hidden_states = self.vision_encoder.execVisionEncoder(input_tensor, batch.num_pending)
            
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(20021, qid, 0, batch.num_pending)
                        
            # TODO: directly batch in the GPU to avoid this GPU to host fetch 
            self.parent.emit_worker.add_to_buffer(vision_embeddings.cpu().detach().numpy(),
                                                vision_second_last_layer_hidden_states.cpu().detach().numpy(),
                                                batch.question_ids,
                                                batch.num_pending)
            
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(20041, qid, 0, 0)
            
            
            

class StepBEmitWorker:
    '''
    This is a batcher for StepB execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.send_buffer = [StepBResultBatchManager() for _ in range(len(self.parent.stepb_next_udl_shards))]  # list of PendingTextDataBatcher
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = False
    
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

    def add_to_buffer(self, vision_embeddings, vision_second_last_layer_hidden_states, question_ids, num_pending):
        '''
        pass by object reference to avoid deep-copy
        '''
        with self.cv:
            for i in range(num_pending):
                shard_pos = question_ids[i] % len(self.parent.stepb_next_udl_shards)
                                
                self.send_buffer[shard_pos].add_result(vision_embeddings[i].view(), 
                                                     vision_second_last_layer_hidden_states[i].view(), 
                                                     question_ids[i].view())
                
               
            self.cv.notify()
            
    def process_and_emit_results(self, to_send):
        for idx, batch_manager in enumerate(to_send):
            if batch_manager.num_queries == 0:
                continue
            
            for qid in batch_manager.question_ids_list[:batch_manager.num_queries]:
                    self.parent.tl.log(20100, qid, 0, 0)
            # serialize the batch_manager
            num_sent = 0
            cur_shard_id = self.parent.stepb_next_udl_shards[idx]
            while num_sent < batch_manager.num_queries:
                serialize_batch_size = min(self.max_emit_batch_size, batch_manager.num_queries - num_sent)
                start_pos = num_sent
                end_pos = num_sent + serialize_batch_size
                for qid in batch_manager.question_ids_list[start_pos:end_pos]:
                    self.parent.tl.log(20030, qid, 0, serialize_batch_size)
                serialized_batch = batch_manager.serialize(start_pos, end_pos)
                for qid in batch_manager.question_ids_list[start_pos:end_pos]:
                    self.parent.tl.log(20031, qid, 0, serialize_batch_size)
                new_key = STEPB_NEXT_UDL_PREFIX + str(self.parent.sent_msg_count)
                self.parent.sent_msg_count += 1

                self.parent.capi.put_nparray(new_key, serialized_batch, 
                                        subgroup_type=STEPB_NEXT_UDL_SUBGROUP_TYPE, 
                                        subgroup_index=STEPB_NEXT_UDL_SUBGROUP_INDEX, 
                                        shard_index=cur_shard_id, 
                                        message_id=0, as_trigger=True, blocking=False) # async put
                
                num_sent += serialize_batch_size
                
            
                
        
    
    def main_loop(self):
        batch_wait_time = self.parent.batch_time_us/1000000
        while self.running:
            to_send = []
            empty = True
            with self.cv:
                for i in range(len(self.send_buffer)):
                    if self.send_buffer[i].num_queries > 0:
                        empty = False
                        break
                    
                if empty:
                    self.cv.wait(timeout=batch_wait_time)
                if not self.running:
                    break
                
                if not empty:
                    to_send = self.send_buffer
                    # Below is shallow copy, to avoid deep copy of the data
                    self.send_buffer = [StepBResultBatchManager() for _ in range(len(self.parent.stepb_next_udl_shards))]
                    
            self.process_and_emit_results(to_send)
            
        


class StepBUDL(UserDefinedLogic):
    '''
    StepBUDL is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepBUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        
        self.checkpoint_path = self.conf["checkpoint_path"]
        self.local_encoder_path = self.conf["local_encoder_path"]
        self.local_projection_path = self.conf["local_projection_path"]
        
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        
        self.stepb_next_udl_shards = self.conf.get("stepb_next_udl_shards", [2])
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
            
        if not self.model_worker:
            self.model_worker = StepBModelWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = StepBEmitWorker(self, 2)
            self.emit_worker.start()
        
    def ocdpo_handler(self,**kwargs):
        '''
        The off-critical data path handler
        '''
        if not self.model_worker:
            self.start_threads()
            
        key = kwargs["key"]
        blob = kwargs["blob"]
        
        received_batch = PixelValueBatcher()
        received_batch.deserialize(blob)
        
        self.model_worker.push_to_pending_batches(received_batch)
        
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"StepBUDL destructor")
        pass
    