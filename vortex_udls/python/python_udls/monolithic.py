import json
import cascade_context
import threading
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from serialize_utils import MonoDataBatcher


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



class Monolithic_UDL(UserDefinedLogic):
    
    def __init__(self,conf_str):
        super(Monolithic_UDL,self).__init__(conf_str)
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
            self.model_worker = StepAModelWorker(self, 1)
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