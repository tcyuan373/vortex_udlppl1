#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import json
import threading
import torch

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from serialize_utils import TextDataBatcher, StepAResultBatchManager, PendingTextDataBatcher
from TextEncoder import TextEncoder



STEPA_NEXT_UDL_PREFIX = "/stepD/resultA_"
STEPA_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
STEPA_NEXT_UDL_SUBGROUP_INDEX = 0

STEPA_WORKER_INITIAL_PENDING_BATCHES = 10

class StepAModelWorker:
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.text_encoder = TextEncoder(self.parent.checkpoint_path, self.parent.local_encoder_path, self.parent.local_projection_path)
        self.pending_batches = [PendingTextDataBatcher(self.max_exe_batch_size) for _ in range(STEPA_WORKER_INITIAL_PENDING_BATCHES)]
        
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


    def push_to_pending_batches(self, text_data_batcher):
        for qid in text_data_batcher.question_ids:
            self.parent.tl.log(10000, qid, 0, 0)
        num_questions = len(text_data_batcher.question_ids)
        question_added = 0
        while question_added < num_questions:
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
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(10030, qid, 0, batch.num_pending)
            # TODO: use direct memory sharing via pointer instead of copying to the host
            # NOTE: use as_tensor instead of torch.LongTensor to avoid a copy
            cur_input_ids = torch.as_tensor(batch.input_ids[:batch.num_pending], dtype=torch.long, device="cuda") 
            cur_attention_mask = torch.as_tensor(batch.attention_mask[:batch.num_pending], dtype=torch.long, device="cuda")
            text_embeddings, text_encoder_hidden_states = self.text_encoder.execTextEncoder(cur_input_ids, cur_attention_mask)
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(10031, qid, 0, batch.num_pending)
            # Appending to the sending buffer to be sent by the emit worker
            self.parent.emit_worker.add_to_buffer(batch.question_ids[:batch.num_pending], 
                                                  batch.text_sequence[:batch.num_pending], 
                                                  batch.input_ids[:batch.num_pending], 
                                                  text_embeddings.cpu().detach().numpy(), 
                                                  text_encoder_hidden_states.cpu().detach().numpy())
            self.pending_batches[self.current_batch].reset()


class StepAEmitWorker:
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.send_buffer = [StepAResultBatchManager() for _ in range(len(self.parent.stepa_next_udl_shards))]  # list of PendingTextDataBatcher
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

    def add_to_buffer(self, question_ids, text_sequence, input_ids,
                      text_embeddings, text_encoder_hidden_states):
        '''
        pass by object reference to avoid deep-copy
        '''
        with self.cv:
            # use question_id to determine which shard to send to
            for i in range(len(question_ids)):
                shard_pos = question_ids[i] % len(self.parent.stepa_next_udl_shards)
                self.send_buffer[shard_pos].add_result(question_ids[i], 
                                                       text_sequence[i], 
                                                       input_ids[i].view(),
                                                       text_embeddings[i].view(),
                                                       text_encoder_hidden_states[i].view())
            self.cv.notify()
            
    def process_and_emit_results(self, to_send):
        for idx, batch_manager in enumerate(to_send):
            if batch_manager.num_queries == 0:
                continue
            # serialize the batch_manager
            num_sent = 0
            cur_shard_id = self.parent.stepa_next_udl_shards[idx]
            
            for qid in batch_manager.question_ids[:batch_manager.num_queries]:
                self.parent.tl.log(10100, qid, 0, batch_manager.num_queries)
            
            while num_sent < batch_manager.num_queries:
                serialize_batch_size = min(self.max_emit_batch_size, batch_manager.num_queries - num_sent)
                start_pos = num_sent
                end_pos = num_sent + serialize_batch_size
                serialized_batch = batch_manager.serialize(start_pos, end_pos)
                new_key = STEPA_NEXT_UDL_PREFIX + str(self.parent.sent_msg_count)
                self.parent.sent_msg_count += 1
                
                self.parent.capi.put_nparray(new_key, serialized_batch, 
                                        subgroup_type=STEPA_NEXT_UDL_SUBGROUP_TYPE, 
                                        subgroup_index=STEPA_NEXT_UDL_SUBGROUP_INDEX, 
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
                    self.send_buffer = [StepAResultBatchManager() for _ in range(len(self.parent.stepa_next_udl_shards))]
                    
            self.process_and_emit_results(to_send)
            
        


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
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        
        self.checkpoint_path = self.conf["checkpoint_path"]
        self.local_encoder_path = self.conf["local_encoder_path"]
        self.local_projection_path = self.conf["local_projection_path"]

        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        
        self.stepa_next_udl_shards = self.conf.get("stepa_next_udl_shards", [2])
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0


    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = StepAModelWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = StepAEmitWorker(self, 2)
            self.emit_worker.start()

    def ocdpo_handler(self,**kwargs):
        '''
        The off-critical data path handler
        '''
        
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
            
        key = kwargs['key']
        blob = kwargs["blob"]
        
        new_batcher = TextDataBatcher()
        new_batcher.deserialize(blob)
        self.model_worker.push_to_pending_batches(new_batcher)
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"StepAUDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()