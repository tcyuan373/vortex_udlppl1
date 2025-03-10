import json
import cascade_context
import threading
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from serialize_utils import MonoDataBatcher, PendingMonoBatcher
from mono_flmr import MONOFLMR

MONO_WORKER_INITIAL_PENDING_BATCHES = 3

class MonoModelWorker:
    '''
    This is a batcher for Monolithic model execution
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.mono_flmr = MONOFLMR(self.parent.index_root_path,
                                  self.parent.index_name,
                                  self.parent.index_experiment_name,
                                  self.parent.checkpoint_path,
                                  self.parent.image_processor_name)
        self.pending_batches = [PendingMonoBatcher(self.max_exe_batch_size) for _ in range(MONO_WORKER_INITIAL_PENDING_BATCHES)]
        
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


    def push_to_pending_batches(self, mono_data_batcher):
        for qid in mono_data_batcher.question_ids:
            self.parent.tl.log(40000, qid, 0, 0)
        
        num_questions = len(mono_data_batcher.question_ids)
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
                    new_batch = PendingMonoBatcher(self.max_batch_size)
                    self.pending_batches.append(new_batch)  
                    free_batch = len(self.pending_batches) - 1
                    space_left = self.pending_batches[free_batch].space_left()
                
                # add as many questions as possible to the pending batch
                self.next_batch = free_batch
                question_start_idx = question_added
                end_idx = self.pending_batches[free_batch].add_data(mono_data_batcher, question_start_idx)
                question_added = end_idx
                #  if we complete filled the buffer, cycle to the next
                if self.pending_batches[free_batch].space_left() == 0:
                    self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    if self.next_batch == self.current_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                        
            self.cv.notify()
            
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
                    # print("found something to process")
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            
            # Execute the batch
            # TODO: use direct memory sharing via pointer instead of copying to the host
            # NOTE: use as_tensor instead of torch.LongTensor to avoid a copy
            
            for qid in cur_question_ids:
                self.parent.tl.log(40031, qid, 0, 0)
            cur_input_ids = batch.input_ids[:batch.num_pending]
            cur_attention_mask = batch.attention_mask[:batch.num_pending]
            cur_pixel_values = batch.pixel_values[:batch.num_pending]
            cur_question_ids = batch.question_ids[:batch.num_pending]
            cur_text_sequence = batch.text_sequence[:batch.num_pending]
            ranking_dict = self.mono_flmr.execFLMR(cur_input_ids, 
                                                   cur_attention_mask,
                                                   cur_pixel_values,
                                                   cur_question_ids,
                                                   cur_text_sequence)
            for qid in cur_question_ids:
                self.parent.tl.log(40100, qid, 0, 0)
            if self.parent.flush_qid in batch.question_ids:
                print(f"StepE finished No.{self.parent.flush_qid} queries")

            self.pending_batches[self.current_batch].reset()
            
            
            

class MonolithicUDL(UserDefinedLogic):
    
    def __init__(self,conf_str):
        super(MonolithicUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        # print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        self.index_root_path        = self.conf["index_root_path"]
        self.index_name             = self.conf["index_name"]
        self.index_experiment_name  = self.conf["index_experiment_name"]
        self.checkpoint_path        = self.conf["checkpoint_path"]
        self.image_processor_name   = self.conf["image_processor_name"]
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.flush_qid = int(self.conf.get("flush_qid", 100))
        self.model_worker = None

        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = MonoModelWorker(self, 1)
            self.model_worker.start()

    def ocdpo_handler(self,**kwargs):
        
        if not self.model_worker:
            self.start_threads()
        key                 = kwargs["key"]
        blob                = kwargs["blob"]
        # bytes_obj           = blob.view(dtype=np.uint8)
        # json_str_decoded    = bytes_obj.decode('utf-8')
        new_batcher = MonoDataBatcher()
        new_batcher.deserialize(blob)
        self.model_worker.push_to_pending_batches(new_batcher)

        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"MonolithicUDL destructor")
        pass