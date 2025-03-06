#!/usr/bin/env python3
import numpy as np
import json
import threading
import torch

import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from TransformerMappingNetwork import MLP, TransformerMappingNetwork
from serialize_utils import (StepCDIntermediateResult, 
                            StepDMessageBatcher,StepBResultBatchManager, 
                            StepAResultBatchManager, PendingStepCDDataBatcher)

STEPD_NEXT_UDL_SHARD_INDEX = 0
STEPCD_NEXT_UDL_SHARDS = [2]
# Initial number of pending batches smaller, because they are allocated on GPU, 
# if the max_exec_batch_size is 16, it takes 3 * 18MB memory on GPU
STEPCD_WORKER_INITIAL_PENDING_BATCHES = 3
MAX_STEPCD_WORKER_PENDING_BATCHES = 8

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
        self.pending_batches = [PendingStepCDDataBatcher(self.max_exe_batch_size) for _ in range(STEPCD_WORKER_INITIAL_PENDING_BATCHES)]

        
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


    def push_to_pending_batches(self, intermediate_result):
        '''
        Adding the intermediate result one-by-one since they are aggregated one-by-one at the UDL side
        '''
        with self.cv:
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
                new_batch = PendingStepCDDataBatcher(self.max_batch_size)
                self.pending_batches.append(new_batch)  
                free_batch = len(self.pending_batches) - 1
                space_left = self.pending_batches[free_batch].space_left()
                
            # add the intermediate result to the pending batch
            self.pending_batches[free_batch].add_data(intermediate_result)
            
            self.next_batch = free_batch
            #  if we complete filled the buffer, cycle to the next
            if self.pending_batches[free_batch].space_left() == 0:
                self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                if self.next_batch == self.current_batch:
                    self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    
            self.cv.notify()
            print("added one query input to queue")
            

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
                    print("StepCD found something to process")
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            
            print("StepCD about to execute")
            
            # Execute the batch
            transformer_mapping_input_features = self.mlp_model.execMLP(batch.vision_second_last_layer_hidden_states[:batch.num_pending,:,:])
            query_embeddings = self.transformer_mapping_model.execTransformerMappingNetwork(
                                batch.input_ids[:batch.num_pending,:], 
                                batch.text_embeddings[:batch.num_pending,:,:], 
                                batch.text_encoder_hidden_states[:batch.num_pending,:,:], 
                                batch.vision_embeddings[:batch.num_pending,:,:],
                                transformer_mapping_input_features)
            result = query_embeddings.cpu().detach().numpy()
            print(f"stepCD finish execution query_emebdding shape: {result.shape}")
            
            # self.parent.emit_worker.add_to_buffer(vision_embeddings.cpu().detach().numpy(),
            #                                     vision_second_last_layer_hidden_states.cpu().detach().numpy(),
            #                                     batch.question_ids,
            #                                     batch.num_pending)
            # print("added to send buffer")
            # self.pending_batches[self.current_batch].reset()


class StepCDEmitWorker:
    '''
    This is a batcher for StepA execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.send_buffer = [StepAResultBatchManager() for _ in range(len(STEPCD_NEXT_UDL_SHARDS))]  # list of PendingTextDataBatcher
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
        pass
            
    def process_and_emit_results(self, to_send):
        pass
        
    
    def main_loop(self):
        pass
            
        


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
        # print(f"ConsolePrinter constructor received json configuration: {self.conf}")
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
        
        # Keep track of collected intermediate results: {query_id0: StepCDIntermediateResult, query_id2:{} ...}
        self.collected_intermediate_results = {}
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        

    def append_result_to_collector(self, key, blob):
        
        step_A_idx = key.find("resultA") 
        step_B_idx = key.find("resultB")
        
        if step_A_idx != -1:
            stepa_serializer = StepAResultBatchManager()
            stepa_serializer.deserialize(blob)
            for idx, qid in enumerate(stepa_serializer.question_ids):
                if not self.collected_intermediate_results.get(qid):
                    self.collected_intermediate_results[qid] = StepCDIntermediateResult()
                # TODO: currently copying, because the execution needs to aggregate results from both stepA and stepB
                #       and the original data may be overwritten in the RDMA buffer
                self.collected_intermediate_results[qid]._question_id = qid
                self.collected_intermediate_results[qid]._np_text_sequence_bytes = np.copy(stepa_serializer.np_text_sequence_bytes[idx])
                self.collected_intermediate_results[qid]._input_ids = torch.Tensor(stepa_serializer.input_ids[idx])
                self.collected_intermediate_results[qid]._text_embeddings = torch.Tensor(stepa_serializer.text_embeds[idx])
                self.collected_intermediate_results[qid]._text_encoder_hidden_states = torch.Tensor(stepa_serializer.text_encoder_hidden_states[idx])
                
                if self.collected_intermediate_results[qid].collected_all():
                    self.model_worker.push_to_pending_batches(self.collected_intermediate_results[qid])
                    self.tl.log(30000, qid, 3, 0)
                    del self.collected_intermediate_results[qid]
                    
            # print("--- StepD has deserialized stepA data ---")
            # stepa_serializer.print_shape()
            
        elif step_B_idx != -1:
            stepb_batcher = StepBResultBatchManager()
            stepb_batcher.deserialize(blob)
            for idx, qid in enumerate(stepb_batcher.question_ids):
                if not self.collected_intermediate_results.get(qid):
                    self.collected_intermediate_results[qid] = StepCDIntermediateResult()
                self.collected_intermediate_results[qid]._question_id = qid
                self.collected_intermediate_results[qid]._vision_embeddings = torch.Tensor(stepb_batcher.vision_embedding[idx])
                self.collected_intermediate_results[qid]._vision_second_last_layer_hidden_states = torch.Tensor(stepb_batcher.vision_second_last_layer_hidden_states[idx])
                self.tl.log(30000, qid, 2, 0)
                
                if self.collected_intermediate_results[qid].collected_all():
                    self.model_worker.push_to_pending_batches(self.collected_intermediate_results[qid])
                    self.tl.log(30000, qid, 3, 0)
                    del self.collected_intermediate_results[qid]
                
            # print("--- StepD has deserialized stepB data ---")
            # stepb_batcher.print_shape()
        
        
    def ocdpo_handler(self, **kwargs):
        # preparing for input : 
            # (input_ids                              # step A      B * seq_len
            # text_embeddings                         # step A      B * 32 * 128
            # text_encoder_hidden_states              # step A      B * 32 * 768
            # vision_embeddings                       # step B      B * 32 * 128
            # vision_second_last_layer_hidden_states)     # step B  B * 256 * 1024
        if not self.model_worker:
            self.model_worker = StepCDModelWorker(self, 0)
            self.model_worker.start()
            self.emit_worker = StepCDEmitWorker(self, 0)
            self.emit_worker.start()
            
        key = kwargs["key"]
        blob = kwargs["blob"]
        self.append_result_to_collector(key, blob)
        
        return
        
        # print(f"Found batch query embeddings of shape: {batch_query_embeddings.shape}")

        self.tl.log(30011, batch_id, 0, 0)
        # self.collected_intermediate_results.erase(batch_id)
        
        # result = {}
        # result['queries'] = self.collected_intermediate_results[batch_id]._queries
        # result['query_embeddings'] = batch_query_embeddings.tolist()
        # result['question_id'] = self.collected_intermediate_results[batch_id]._question_id
        # res_json_str = json.dumps(result)
        # res_json_byte = res_json_str.encode('utf-8')
        
        stepd_batcher = StepDMessageBatcher()
        stepd_batcher.queries = self.collected_intermediate_results[batch_id]._queries
        stepd_batcher.query_embeddings = batch_query_embeddings.cpu().detach().numpy()
        stepd_batcher.question_ids = self.collected_intermediate_results[batch_id]._question_id
        stepd_batcher_np = stepd_batcher.serialize()
        
        subgroup_type = "VolatileCascadeStoreWithStringKey"
        subgroup_index = 0
        
        res = self.capi.put(f"/stepE/stepD_{batch_id}", stepd_batcher_np.tobytes(), subgroup_type=subgroup_type,
                subgroup_index=subgroup_index,shard_index=STEPD_NEXT_UDL_SHARD_INDEX, message_id=1, as_trigger=True, blocking=False)

        self.tl.log(30020, batch_id, 0, 0)
        
        if batch_id == 49:
            self.tl.flush(f"node{self.my_id}_STEPD_udls_timestamp.dat")
            print("StepD TL flushed!!!")
        # garbage cleaning via emit and del
        del self.collected_intermediate_results[batch_id]
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass