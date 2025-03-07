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
                    new_batch = PendingSearchBatcher(self.max_batch_size)
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
            print("stepE added to queue")
            

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
                    print("stepE found something to process")
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            
            # Execute the batch
            # TODO: use direct memory sharing via pointer instead of copying to the host
            queries = dict(zip(batch.question_ids, batch.text_sequence))
            rank_dict = self.text_encoder.process_search(queries, batch.query_embeddings[:batch.num_pending])
            print(f"~~~~ Finished StepE processed {batch.num_pending} queries ~~~~")
            print(f"Ranking: {rank_dict.keys()}")
            


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
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" #set with 1 for the use of cuda:1, and if set to be "0,1", all two gpus will be used

        if not self.model_worker:
            self.start_threads()
            
        new_batcher = StepDMessageBatcher()
        new_batcher.deserialize(blob)
        self.model_worker.push_to_pending_batches(new_batcher)
        return
        
        # bytes_obj           = blob.tobytes()
        # json_str_decoded    = bytes_obj.decode('utf-8')
        # cluster_result      = json.loads(json_str_decoded)
        queries_texts       = cluster_result['queries']
        query_embeddings    = cluster_result['query_embeddings']
        question_ids        = cluster_result['question_ids']
        bsize               = None   #using the best perf config, same as the FLMR paper setting
        
        
        # queries = {question_id[i]: queries_texts[i] for i in range(len(queries_texts))}
        assert len(queries_texts) == len(question_ids)
        queries = dict(zip(question_ids, queries_texts))
        
        step_D_idx = key.find('stepD')
        uds_idx = key.find("_")
        batch_id = int(key[uds_idx+1:])
        
        
        self.tl.log(40000, batch_id, 0, 0)
        ranking = self.process_search(queries, query_embeddings, bsize)
        
        # if not self.collected_intermediate_results.get(batch_id):
        #     self.collected_intermediate_results[batch_id] = IntermediateResult()
            
        # if step_D_idx != -1:
        #     self.collected_intermediate_results[batch_id]._queries = queries
        #     self.collected_intermediate_results[batch_id]._query_embeddings = query_embeddings
            
        # if not self.collected_intermediate_results[batch_id].has_all():
        #     return
            
        # ranking = self.process_search(self.collected_intermediate_results[batch_id]._queries, 
        #                               self.collected_intermediate_results[batch_id]._query_embeddings,
        #                               bsize)

        # print('==========Finished Searching==========')
        # print(f"Got queries: {self.collected_intermediate_results[batch_id]._queries}")
        # print(f'Got a ranking dictionary for batch {batch_id}: {ranking}')
        self.tl.log(40100, batch_id, 0, 0)
        
        # erase the batch id dict{} 
        # del self.collected_intermediate_results[batch_id]
        if batch_id == 49:
            self.tl.flush(f"node{self.my_id}_STEPE_udls_timestamp.dat")
            print("Time Log Flushed!!!")
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"StepEUDL destructor")
        pass