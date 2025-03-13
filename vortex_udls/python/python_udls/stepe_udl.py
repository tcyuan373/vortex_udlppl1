#!/usr/bin/env python3
import json
import threading
import torch
import os
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from stepe_search import StepESearch
from serialize_utils import StepDMessageBatcher, PendingSearchBatcher

STEPE_WORKER_INITIAL_PENDING_BATCHES = 3

class StepEModelWorker:
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.text_encoder = StepESearch(self.parent.index_root_path, self.parent.index_experiment_name, self.parent.index_name)
        # PendingSearchBatcher creates a batch of embeddings on CUDA
        self.pending_batches = [PendingSearchBatcher(self.max_exe_batch_size) for _ in range(STEPE_WORKER_INITIAL_PENDING_BATCHES)]
        
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
            self.parent.tl.log(40000, qid, self.next_batch, text_data_batcher.num_queries)
            if qid == self.parent.flush_qid:
                print(f"StepE received No.{qid} queries")
            
            
        num_questions = len(text_data_batcher.question_ids)
        question_added = 0
        while question_added < num_questions:
            with self.cv:
                # Block if no space available at the pending queue
                while True:
                    if not self.new_space_available:
                        self.parent.tl.log(40021, text_data_batcher.question_ids[question_added],0, 0)
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
                    # if free_batch >= self.next_batch:
                    if free_batch == initial_batch:
                        break
                    space_left = self.pending_batches[free_batch].space_left()
                if space_left != 0:
                    # add as many questions as possible to the pending batch
                    self.next_batch = free_batch
                    question_start_idx = question_added
                    end_idx = self.pending_batches[free_batch].add_data(text_data_batcher, question_start_idx)
                    question_added = end_idx
                    for qid in text_data_batcher.question_ids[question_start_idx:end_idx]:
                        self.parent.tl.log(40001, qid, free_batch, self.pending_batches[self.next_batch].num_pending)
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
                    for qid in batch.question_ids[:batch.num_pending]:
                        self.parent.tl.log(40011, qid, 0, batch.num_pending)
                        
                    if self.current_batch == self.next_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches) 
                    self.new_space_available = True
                    self.cv.notify()
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(40030, qid, 0, batch.num_pending)
            # Execute the batch
            # TODO: use direct memory sharing via pointer instead of copying to the host 
            queries = dict(zip(batch.question_ids, batch.text_sequence))
            rank_dict = self.text_encoder.process_search(queries, batch.query_embeddings[:batch.num_pending])
            
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(40031, qid, 0, batch.num_pending)
            
            if self.parent.flush_qid in batch.question_ids:
                print(f"StepE finished No.{self.parent.flush_qid} queries")
            
            # self.parent.capi.put_nparray("finish", np.array(batch.question_ids), subgroup_type=0, subgroup_index=0, shard_index=0, message_id=1)
            


class StepEUDL(UserDefinedLogic):
    '''
    StepEUDL is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepEUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        self.index_root_path = self.conf["index_root_path"]
        self.index_experiment_name = self.conf["index_experiment_name"]
        self.index_name = self.conf["index_name"]
        self.max_exe_batch_size = self.conf["max_exe_batch_size"]
        self.batch_time_us = self.conf["batch_time_us"]
        self.flush_qid = self.conf["flush_qid"]
        self.model_worker = None
    
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = StepEModelWorker(self, 1)
            self.model_worker.start()
            
            
    def ocdpo_handler(self,**kwargs):
        key                 = kwargs["key"]
        blob                = kwargs["blob"]

        if not self.model_worker:
            self.start_threads()
            
        new_batcher = StepDMessageBatcher()
        new_batcher.deserialize(blob)
        self.model_worker.push_to_pending_batches(new_batcher)
        
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"StepEUDL destructor")
        pass