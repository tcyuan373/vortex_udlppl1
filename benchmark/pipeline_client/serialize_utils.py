import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import copy

'''
Serialization helper classes and functions for Pipeline1, FLMR pipeline 
'''


def utf8_length(s: str) -> int:
    """Computes the length of a UTF-8 encoded string without actually encoding it."""
    return sum(1 + (ord(c) >= 0x80) + (ord(c) >= 0x800) + (ord(c) >= 0x10000) for c in s)



# ===============  Class and Serializer for Monolithic pipeline ===============

class MonoDataBatcher:
    def __init__(self):
        # All fields must have the same batch size.
        self.pixel_values = None        # np.ndarray of shape (batch_size, 1, 224, 224) and dtype=np.float32
        self.text_sequence = []         # List of strings (one per query)
        self.question_ids = []          # List of ints (one per query)
        self.input_ids = None           # np.ndarray of shape (batch_size, 32) and dtype=np.int64
        self.attention_mask = None      # np.ndarray of shape (batch_size, 32) and dtype=np.int64   
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
    
    
    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous buffer:
          - Header: 4 bytes for batch_size (uint32).
          - Metadata: For each text_sequence element, store two int64 values (offset and length).
          - Fixed segments:
              * question_ids: (batch_size,) int64.
              * input_ids: (batch_size, 32) int64.
              * attention_mask: (batch_size, 32) int64.
              * pixel_values: (batch_size, 1, 224, 224) float32.
          - Variable segment:
              * text_sequence: concatenated UTF-8 encoded bytes.
        """
        batch_size = len(self.question_ids)
        # Compute offsets and total size for text_sequence without storing encoded values.
        text_seq_offsets = []
        offset_temp = 0
        for t in self.text_sequence:
            text_seq_offsets.append(offset_temp)
            offset_temp += utf8_length(t)
        total_text_seq_size = offset_temp
        
        header_size = np.dtype(np.uint32).itemsize   # 4 bytes
        
        # Define metadata: two int64 values per example.
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        
        # Fixed segments sizes.
        question_ids_size = batch_size * np.dtype(np.int64).itemsize
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        attention_mask_size = batch_size * 32 * np.dtype(np.int64).itemsize
        pixel_values_size = batch_size * 1  * 3 * 224 * 224 * np.dtype(np.float32).itemsize
        
        total_size = (header_size + metadata_size + question_ids_size +
                        input_ids_size + attention_mask_size + pixel_values_size + total_text_seq_size)
        
        # Allocate one contiguous buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        offset = 0
        
        # --- Write header: batch_size (uint32) ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        offset += header_size
        
        # --- Write metadata for text_sequence ---
        metadata_start = offset
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size],
                                       dtype=metadata_dtype)
        for i, t in enumerate(self.text_sequence):
            metadata_array[i]["text_sequence_offset"] = text_seq_offsets[i]
            metadata_array[i]["text_sequence_length"] = utf8_length(t)
        offset += metadata_size
        
        # --- Write question_ids ---
        qids_start = offset
        qids_array = np.frombuffer(buffer[qids_start:qids_start+question_ids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        offset += question_ids_size
        
        # --- Write input_ids ---
        input_ids_start = offset
        buffer[input_ids_start:input_ids_start+input_ids_size] = self.input_ids.view(np.uint8).reshape(-1)
        offset += input_ids_size
        
        # --- Write attention_mask ---
        attention_mask_start = offset
        buffer[attention_mask_start:attention_mask_start+attention_mask_size] = self.attention_mask.view(np.uint8).reshape(-1)
        offset += attention_mask_size
        
        # --- Write pixel_values ---
        pixel_values_start = offset
        buffer[pixel_values_start:pixel_values_start+pixel_values_size] = self.pixel_values.view(np.uint8).reshape(-1)
        offset += pixel_values_size
        
        # --- Write text_sequence segment ---
        # Instead of pre-encoding, encode each string inline.
        text_seq_start = offset
        pos = text_seq_start
        for t in self.text_sequence:
            enc = t.encode("utf-8")  # Encode inline.
            n = len(enc)             # Alternatively, self.utf8_length(t)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        offset += total_text_seq_size
        
        self._bytes = buffer
        return buffer
    
    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous buffer back into the fields.
        Layout:
          - Header: 4 bytes (batch_size, uint32)
          - Metadata: per example text_sequence offset and length (int64)
          - question_ids: array of int64, shape (batch_size,)
          - input_ids: array of int64, shape (batch_size, 32)
          - attention_mask: array of int64, shape (batch_size, 32)
          - pixel_values: array of float32, shape (batch_size, 1, 224, 224)
          - text_sequence: concatenated UTF-8 bytes.
        After deserialization, question_ids is converted to a Python list.
        """
        self._bytes = data
        buffer = data
        offset = 0
        
        # --- Read header: batch_size ---
        header_size = np.dtype(np.uint32).itemsize
        batch_size = int(np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0])
        offset += header_size
        
        # --- Read metadata for text_sequence ---
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer, dtype=metadata_dtype, count=batch_size, offset=offset)
        offset += metadata_size
        
        # --- Read question_ids ---
        question_ids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset).tolist()
        offset += question_ids_size
        
        # --- Read input_ids ---
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        num_input_ids = batch_size * 32
        input_ids = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += input_ids_size
        
        # --- Read attention_mask ---
        attention_mask_size = batch_size * 32 * np.dtype(np.int64).itemsize
        attention_mask = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += attention_mask_size
        
        # --- Read pixel_values ---
        pixel_values_size = batch_size * 1 * 3 *  224 * 224 * np.dtype(np.float32).itemsize
        pixel_values = np.frombuffer(buffer, dtype=np.float32, count=batch_size * 1 * 3* 224 * 224, offset=offset).reshape((batch_size, 1, 3, 224, 224))
        offset += pixel_values_size
        
        # --- Read text_sequence segment ---
        total_text_seq_size = sum(int(x) for x in metadata_array["text_sequence_length"])
        text_seq_bytes = buffer[offset:offset+total_text_seq_size]
        offset += total_text_seq_size
        text_sequence = []
        for m in metadata_array:
            start = int(m["text_sequence_offset"])
            length = int(m["text_sequence_length"])
            text_sequence.append(text_seq_bytes[start:start+length].tobytes().decode("utf-8"))
            
        # Restore fields.
        self.question_ids = qids           # as a Python list.
        self.text_sequence = text_sequence
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.pixel_values = pixel_values
         
         
         
class PendingMonoBatcher:
    '''
    Super batch of MonoDataBatcher, used for MonolithicModel runtime batch execution
    '''
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        # TODO: when using GPU RDMA direct, below fields should be allocated in CUDA memory
        self.question_ids = []      # List[int] of length batch_size.
        self.text_sequence = []     # List[str] of length batch_size.
        self.input_ids = torch.empty((self.max_batch_size, 32), dtype=torch.int64, device="cuda")
        self.attention_mask = torch.empty((self.max_batch_size, 32), dtype=torch.int64, device="cuda")
        self.pixel_values = torch.empty((self.max_batch_size, 1, 3, 224, 224), dtype=torch.float32, device="cuda")
        
    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, MonoDataBatcher, start_pos):
        num_to_add = min(self.space_left(), len(MonoDataBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(copy.deepcopy(MonoDataBatcher.question_ids[start_pos:end_pos]))
        self.text_sequence.extend(copy.deepcopy(MonoDataBatcher.text_sequence[start_pos:end_pos]))
        pending_end_pos = self.num_pending + num_to_add
        self.input_ids[self.num_pending:pending_end_pos].copy_(torch.tensor(MonoDataBatcher.input_ids[start_pos:end_pos], device="cuda"))
        self.attention_mask[self.num_pending:pending_end_pos].copy_(torch.tensor(MonoDataBatcher.attention_mask[start_pos:end_pos], device="cuda"))
        self.pixel_values[self.num_pending:pending_end_pos].copy_(torch.tensor(MonoDataBatcher.pixel_values[start_pos:end_pos], device="cuda"))
        self.num_pending = pending_end_pos
        return end_pos
    
    def reset(self):
        self.question_ids = []
        self.text_sequence = []
        self.input_ids.fill_(0)
        self.attention_mask.fill_(0)
        self.pixel_values.fill_(0)
        self.num_pending = 0
        

#===============  Class and Serializer for Step A Text encoder ===============

class TextDataBatcher:
    '''
    batcher for client to send a batch of text to stepA UDL 
    '''
    def __init__(self):
        # All fields must have the same batch size.
        self.question_ids = []      # List[int] of length batch_size.
        self.text_sequence = []     # List[str] of length batch_size.
        self.input_ids = None       # np.ndarray of shape (batch_size, 32), dtype=np.int64.
        self.attention_mask = None  # np.ndarray of shape (batch_size, 32), dtype=np.int64.
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)

    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous buffer:
          - Header: 4 bytes for batch_size (uint32).
          - Metadata: For each text_sequence element, store two int64 values (offset and length).
          - Fixed segments:
              * question_ids: (batch_size,) int64.
              * input_ids: (batch_size, 32) int64.
              * attention_mask: (batch_size, 32) int64.
          - Variable segment:
              * text_sequence: concatenated UTF-8 encoded bytes.
        """
        batch_size = len(self.question_ids)
        if len(self.text_sequence) != batch_size or self.input_ids.shape[0] != batch_size or self.attention_mask.shape[0] != batch_size:
            raise ValueError("TextDataBatcher input dimension mismatch.")
        
        # Compute offsets and total size for text_sequence without storing encoded values.
        text_seq_offsets = []
        offset_temp = 0
        for t in self.text_sequence:
            text_seq_offsets.append(offset_temp)
            offset_temp += utf8_length(t)
        total_text_seq_size = offset_temp

        header_size = np.dtype(np.uint32).itemsize   # 4 bytes
        
        # Define metadata: two int64 values per example.
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        
        # Fixed segments sizes.
        question_ids_size = batch_size * np.dtype(np.int64).itemsize
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        attention_mask_size = batch_size * 32 * np.dtype(np.int64).itemsize
        
        total_size = (header_size + metadata_size + question_ids_size +
                      input_ids_size + attention_mask_size + total_text_seq_size)
        
        # Allocate one contiguous buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        offset = 0

        # --- Write header: batch_size (uint32) ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        offset += header_size

        # --- Write metadata for text_sequence ---
        metadata_start = offset
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size],
                                       dtype=metadata_dtype)
        for i, t in enumerate(self.text_sequence):
            metadata_array[i]["text_sequence_offset"] = text_seq_offsets[i]
            metadata_array[i]["text_sequence_length"] = utf8_length(t)
        offset += metadata_size

        # --- Write question_ids ---
        qids_start = offset
        qids_array = np.frombuffer(buffer[qids_start:qids_start+question_ids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        offset += question_ids_size

        # --- Write input_ids ---
        input_ids_start = offset
        buffer[input_ids_start:input_ids_start+input_ids_size] = self.input_ids.view(np.uint8).reshape(-1)
        offset += input_ids_size

        # --- Write attention_mask ---
        attention_mask_start = offset
        buffer[attention_mask_start:attention_mask_start+attention_mask_size] = self.attention_mask.view(np.uint8).reshape(-1)
        offset += attention_mask_size

        # --- Write text_sequence segment ---
        # Instead of pre-encoding, encode each string inline.
        text_seq_start = offset
        pos = text_seq_start
        for t in self.text_sequence:
            enc = t.encode("utf-8")  # Encode inline.
            n = len(enc)             # Alternatively, self.utf8_length(t)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        offset += total_text_seq_size

        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous buffer back into the fields.
        Layout:
          - Header: 4 bytes (batch_size, uint32)
          - Metadata: per example text_sequence offset and length (int64)
          - question_ids: array of int64, shape (batch_size,)
          - input_ids: array of int64, shape (batch_size, 32)
          - attention_mask: array of int64, shape (batch_size, 32)
          - text_sequence: concatenated UTF-8 bytes.
        After deserialization, question_ids is converted to a Python list.
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header: batch_size ---
        header_size = np.dtype(np.uint32).itemsize
        batch_size = int(np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0])
        offset += header_size

        # --- Read metadata for text_sequence ---
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer, dtype=metadata_dtype, count=batch_size, offset=offset)
        offset += metadata_size

        # --- Read question_ids ---
        question_ids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset).tolist()
        offset += question_ids_size

        # --- Read input_ids ---
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        num_input_ids = batch_size * 32
        input_ids = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += input_ids_size

        # --- Read attention_mask ---
        attention_mask_size = batch_size * 32 * np.dtype(np.int64).itemsize
        attention_mask = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += attention_mask_size

        # --- Read text_sequence segment ---
        total_text_seq_size = sum(int(x) for x in metadata_array["text_sequence_length"])
        text_seq_bytes = buffer[offset:offset+total_text_seq_size]
        offset += total_text_seq_size
        text_sequence = []
        for m in metadata_array:
            start = int(m["text_sequence_offset"])
            length = int(m["text_sequence_length"])
            text_sequence.append(text_seq_bytes[start:start+length].tobytes().decode("utf-8"))

        # Restore fields.
        self.question_ids = qids           # as a Python list.
        self.text_sequence = text_sequence
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        


class PendingTextDataBatcher():
    '''
    Super batch of TextDataBatcher, used for model runtime batch execution
    '''
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        # TODO: when using GPU RDMA direct, below fields should be allocated in CUDA memory
        self.question_ids = []      # List[int] of length batch_size.
        self.text_sequence = []     # List[str] of length batch_size.
        self.input_ids = np.empty((self.max_batch_size, 32), dtype=np.int64)
        self.attention_mask = np.empty((self.max_batch_size, 32), dtype=np.int64)

    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, TextDataBatcher, start_pos):
        num_to_add = min(self.space_left(), len(TextDataBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(TextDataBatcher.question_ids[start_pos:end_pos])
        self.text_sequence.extend(TextDataBatcher.text_sequence[start_pos:end_pos])
        pending_end_pos = self.num_pending + num_to_add
        self.input_ids[self.num_pending:pending_end_pos] = TextDataBatcher.input_ids[start_pos:end_pos]
        self.attention_mask[self.num_pending:pending_end_pos] = TextDataBatcher.attention_mask[start_pos:end_pos]
        self.num_pending = pending_end_pos
        return end_pos

    def reset(self):
        '''
        Reset the fields
        @attention_mask are inputs to the encoder execution, and no longer needed afterwards, setting them to zeros during reset
        @input_ids @question_ids and @text_sequence are still needed to form stepAResultBatcher
         reassigned them to new list and the reference counter will now be held by the new object StepAResultBatcher
        '''
        self.question_ids = []
        self.text_sequence = []
        self.input_ids = np.empty((self.max_batch_size, 32), dtype=np.int64)
        self.attention_mask.fill(0)
        self.num_pending = 0
        

    
class StepAResultBatchManager:
    '''
    Batcher to send to the next UDL
    '''
    def __init__(self):
        self.num_queries = 0
        self.question_ids = []      # List[int] of length batch_size.
        
        # Variables used for serialize
        self.text_sequence = []     # List[str] of length batch_size.
        self.input_ids_list = []       # list of (np.ndarray of shape (1, 32), dtype=np.int64).
        self.text_embeds_list = []     # list of (np.ndarray of shape (1, 32, 128), dtype=np.float32).
        self.text_encoder_hidden_states_list = []  # list of (np.ndarray of shape (1, 32, 768), dtype=np.float32).
        self.text_pos = {}   # qid -> (offset, length)
        
        # Variables used for deserialize
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        self.np_text_sequence_bytes = []     # List of np.ndarray that is utf-8 encoded text_sequence, only decode it when asked
        self.input_ids = None       # np.ndarray of shape (batch_size, 32), dtype=np.int64.
        self.text_embeds = None     # np.ndarray of shape (batch_size, 32, 128), dtype=np.float32.
        self.text_encoder_hidden_states = None  # np.ndarray of shape (batch_size, 32, 768), dtype=np.float32.
        
        # # metadata sizes used for serialization and deserialization
        self.header_size = np.dtype(np.uint32).itemsize
        self.metadata_dtype = np.dtype([("query_offset", np.int64), ("query_length", np.int64)])
        
        

    def add_result(self, question_id, text_sequence, input_ids, text_embeddings, text_encoder_hidden_states):
        self.question_ids.append(question_id)
        self.text_sequence.append(text_sequence)
        self.input_ids_list.append(input_ids)
        self.text_embeds_list.append(text_embeddings)
        self.text_encoder_hidden_states_list.append(text_encoder_hidden_states)
        self.num_queries += 1
        


    def serialize(self, start_pos, end_pos):
        """
        Serializes the aggregated fields from self.stepa_results into one contiguous byte buffer.
        The serialized format is as follows:
          - Header: 4 bytes (batch_size, as uint32).
          - Metadata for queries: for each query, two int64 values:
                * query_offset: starting offset (in bytes) within the variable segment.
                * query_length: byte-length of the query.
          - Fixed segments:
                * question_ids: (batch_size,) int64.
                * input_ids: (batch_size, 32) int64.
                * text_embeds: (batch_size, 32, 128) float32.
                * text_encoder_hidden_states: (batch_size, 32, 768) float32.
          - Variable segment:
                * queries: concatenated UTF-8 encoded bytes.
        @Note: this method serialize the data from stepa_results via slicing from the stepa_results_start_end_pos. 
               This is due to the max_emit_batch_size limit for the message passing
        """
        # Compute the metadata position and total size
        batch_size = end_pos - start_pos
        metadata_size = batch_size * self.metadata_dtype.itemsize
        qids_size = batch_size * np.dtype(np.int64).itemsize
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        text_embeds_size = batch_size * 32 * 128 * np.dtype(np.float32).itemsize
        hidden_states_size = batch_size * 32 * 768 * np.dtype(np.float32).itemsize
        text_offset = self.header_size + metadata_size + qids_size + input_ids_size + text_embeds_size + hidden_states_size
        total_text_sequence_size = 0
        cur_text_offset = text_offset
        for idx in range(start_pos, end_pos):
            text_seq_size = utf8_length(self.text_sequence[idx])
            self.text_pos[self.question_ids[idx]] = (cur_text_offset, text_seq_size)
            cur_text_offset += text_seq_size
            total_text_sequence_size += text_seq_size
        total_size = text_offset + total_text_sequence_size

        # TODO: use blob generator to generate Blob object inplace
        # Allocate one contiguous buffer.
        serialized_buffer = np.empty(total_size, dtype=np.uint8)
        
        # Determine segment positions     
        metadata_pos = self.header_size
        qids_offset = metadata_pos + metadata_size
        input_ids_offset = qids_offset + qids_size
        text_embeds_offset = input_ids_offset + input_ids_size
        hidden_states_offset = text_embeds_offset + text_embeds_size
        
        # Write data into the serialized_buffer buffer
        # Get the array reference to write
        metadata_array = np.frombuffer(serialized_buffer[metadata_pos:metadata_pos + metadata_size], 
                                       dtype=self.metadata_dtype)
        qid_array = np.frombuffer(serialized_buffer[qids_offset:qids_offset + qids_size], 
                                  dtype=np.int64)
        input_ids_array = np.frombuffer(serialized_buffer[input_ids_offset:input_ids_offset + input_ids_size], 
                                        dtype=np.int64).reshape((batch_size, 32))
        text_embeds_array = np.frombuffer(serialized_buffer[text_embeds_offset:text_embeds_offset + text_embeds_size], 
                                          dtype=np.float32).reshape((batch_size, 32, 128))
        hidden_states_array = np.frombuffer(serialized_buffer[hidden_states_offset:hidden_states_offset + hidden_states_size], 
                                            dtype=np.float32).reshape((batch_size, 32, 768))
        
        # Write Header: batch_size (as uint32)
        np.frombuffer(serialized_buffer[:self.header_size], dtype=np.uint32)[0] = batch_size
        written_counter = 0  # local position in this byte buffer
        for manager_idx in range(start_pos, end_pos):
            qid = self.question_ids[manager_idx]
            abs_offset, qlen = self.text_pos[qid]
            metadata_array[written_counter]["query_offset"] = abs_offset  # Absolute offset in the entire buffer.
            metadata_array[written_counter]["query_length"] = qlen
            
            qid_array[written_counter] = qid
            input_ids_array[written_counter] = self.input_ids_list[manager_idx]
            text_embeds_array[written_counter] = self.text_embeds_list[manager_idx]
            hidden_states_array[written_counter] = self.text_encoder_hidden_states_list[manager_idx]
            # Below method allocate memory for the text sequence to encode it, and then copy it to the _bytes array, which isn't ideal with this memory allocation and copy
            # TODO: better way to write to _bytes with less copy, constraint by python
            serialized_buffer[abs_offset:abs_offset + qlen] = np.frombuffer(self.text_sequence[manager_idx].encode("utf-8"), dtype=np.uint8)
            written_counter += 1
        return serialized_buffer
                

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous buffer back into the original fields.
        Expected layout (in order):
          - Header: 4 bytes (uint32: batch_size)
          - Metadata for queries: per query (query_offset and query_length as int64)
          - question_ids: array of int64, shape (batch_size,)
          - input_ids: array of int64, shape (batch_size, 32)
          - text_embeds: array of float32, shape (batch_size, 32, 128)
          - text_encoder_hidden_states: array of float32, shape (batch_size, 32, 768)
          - queries: concatenated UTF-8 bytes.
        After deserialization, question_ids is converted to a Python list.
        """
        # # NOTE: need to create copy here to avoid buffer been overwriten, due to the aggregating mechanism required at this step
        # # fixed already
        # buffer = data.copy()
        offset = 0

        # --- Read header ---
        batch_size = int(np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0])
        offset += self.header_size

        # --- Read metadata for queries ---
        metadata_size = batch_size * self.metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer, dtype=self.metadata_dtype, count=batch_size, offset=offset)
        offset += metadata_size

        # --- Read question_ids ---
        qids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset).tolist()
        offset += qids_size

        # --- Read input_ids ---
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        num_input_ids = batch_size * 32
        input_ids = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += input_ids_size

        # --- Read text_embeds ---
        text_embeds_size = batch_size * 32 * 128 * np.dtype(np.float32).itemsize
        num_text_embeds = batch_size * 32 * 128
        text_embeds = np.frombuffer(buffer, dtype=np.float32, count=num_text_embeds, offset=offset).reshape((batch_size, 32, 128))
        offset += text_embeds_size

        # --- Read text_encoder_hidden_states ---
        hidden_states_size = batch_size * 32 * 768 * np.dtype(np.float32).itemsize
        num_hidden_states = batch_size * 32 * 768
        text_encoder_hidden_states = np.frombuffer(buffer, dtype=np.float32, count=num_hidden_states, offset=offset).reshape((batch_size, 32, 768))
        offset += hidden_states_size

        # --- Read queries variable segment ---
        for m in metadata_array:
            start = int(m["query_offset"])
            length = int(m["query_length"])
            # Extract query bytes from the entire buffer using the absolute offset.
            query_bytes = buffer[start:start+length]
            # text_sequence.append(query_bytes.tobytes().decode("utf-8"))
            self.np_text_sequence_bytes.append(query_bytes)

        # Restore fields.
        self.question_ids = qids
        self.input_ids = input_ids
        self.text_embeds = text_embeds
        self.text_encoder_hidden_states = text_encoder_hidden_states
        # self.decode_text_sequence()  # Only decode the text sequence when needed
    
    def decode_text_sequence(self):
        for idx in range(len(self.np_text_sequence_bytes)):
            self.text_sequence.append(self.np_text_sequence_bytes[idx].tobytes().decode("utf-8"))
            
        
    def print_shape(self):
        print("--- StepAMessageResultBatcher shape info ---")
        print(f"question_ids: {len(self.question_ids)}")
        print(f"input_ids: {self.input_ids.shape}")
        print(f"text_embeds: {self.text_embeds.shape}")
        print(f"text_encoder_hidden_states: {self.text_encoder_hidden_states.shape}")
        self.decode_text_sequence()
        print(f"text_sequence: {len(self.text_sequence)}")



# ===============  Class and Serializer for Step B Image encoder ===============

class PixelValueBatcher:
    '''
    Batcher for client to send a batch of pixel values to stepB UDL
    '''
    def __init__(self):
        # Expected shape: (batch_size, 1, 3, 224, 224) and dtype=np.float32.
        self.pixel_values: np.ndarray = None
        # Query IDs as a NumPy array of shape (batch_size,) and dtype=np.int64.
        self.question_ids: np.ndarray = None
        # Serialized bytes buffer.
        self._bytes: np.ndarray = None

    def serialize(self) -> np.ndarray:
        """
        Serialize the pixel_values and question_ids into a contiguous buffer.
        Layout:
        [4 bytes: batch_size (uint32)] | [question_ids (int64)] | [pixel_values (float32)]
        """
        if self.pixel_values is None or self.question_ids is None:
            raise ValueError("Both pixel_values and question_ids must be provided.")
        batch_size = self.question_ids.shape[0]
        if self.pixel_values.shape[0] != batch_size:
            raise ValueError("The first dimension of pixel_values must match the number of question_ids.")
        
        # Calculate sizes.
        header_size = np.dtype(np.uint32).itemsize   # 4 bytes
        question_ids_size = self.question_ids.nbytes   # batch_size * 8 bytes
        pixel_values_size = self.pixel_values.nbytes  # batch_size * 1 * 3 * 224 * 224 * 4
        total_size = header_size + question_ids_size + pixel_values_size
        
        # Allocate one contiguous buffer.
        buffer = np.empty(total_size, dtype=np.uint8)
        
        # Write to buffer
        offset = 0
        header_arr = np.array([batch_size], dtype=np.uint32)
        buffer[offset:offset + header_size] = header_arr.view(np.uint8)
        offset += header_size
        buffer[offset:offset + question_ids_size] = self.question_ids.view(np.uint8)
        offset += question_ids_size
        buffer[offset:offset + pixel_values_size] = self.pixel_values.view(np.uint8).reshape(-1)
        offset += pixel_values_size
        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserialize a contiguous uint8 buffer into question_ids and pixel_values with zero-copy.
        The layout of the buffer is:
        [4 bytes: batch_size (uint32)] | [question_ids (int64)] | [pixel_values (float32)]
        where pixel_values is expected to have shape (batch_size, 1, 3, 224, 224).
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header: batch_size ---
        header_size = np.dtype(np.uint32).itemsize  # 4 bytes
        batch_size = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += header_size
        
        self.question_ids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset).reshape(batch_size)    
        offset += self.question_ids.nbytes

        pixel_shape = (batch_size, 1, 3, 224, 224)
        num_pixels = np.prod(pixel_shape)
        self.pixel_values = np.frombuffer(buffer, dtype=np.float32, count=num_pixels, offset=offset).reshape(pixel_shape)
        

class PendingVisionDataBatcher():
    '''
    Super batch of PixelValueBatcher, used for model runtime batch execution
    '''
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        # TODO: when using GPU RDMA direct, below fields should be allocated in CUDA memory
        self.pixel_values = np.empty((self.max_batch_size, 1, 3, 224, 224), dtype=np.float32)
        self.question_ids = np.empty((self.max_batch_size,), dtype=np.int64)

    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, PixelValueBatcher, start_pos):
        num_to_add = min(self.space_left(), PixelValueBatcher.question_ids.shape[0] - start_pos)
        end_pos = start_pos + num_to_add
        self.pixel_values[self.num_pending:self.num_pending + num_to_add] = PixelValueBatcher.pixel_values[start_pos:end_pos]
        self.question_ids[self.num_pending:self.num_pending + num_to_add] = PixelValueBatcher.question_ids[start_pos:end_pos]
        self.num_pending += num_to_add
        return end_pos
    
    def reset(self):
        '''
        Reset the fields
        '''
        self.pixel_values = np.empty((self.max_batch_size, 1, 3, 224, 224), dtype=np.float32)
        self.question_ids = np.empty((self.max_batch_size,), dtype=np.int64)
        self.num_pending = 0

   
class StepBResultBatchManager:
    '''
    Batcher to send to the next UDL
    '''
    def __init__(self):
        self.num_queries = 0
        
        # Variables used for serialize
        self.question_ids_list = []      # List[int] of (np.ndarray of shape (1,), dtype=np.int64)
        self.vision_embedding_list = []  # List[np.ndarray] of (np.ndarray of shape (1, 32, 128), dtype=np.float32)
        self.vision_second_last_layer_hidden_states_list = [] # List[np.ndarray] of (np.ndarray of shape (1, 256, 1024), dtype=np.float32)
        
        # Variables used for deserialize
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        self.question_ids: np.ndarray = None
        self.vision_embedding: np.ndarray = None    # np.ndarray of shape (batch_size, 32, 128), dtype=np.float32.
        self.vision_second_last_layer_hidden_states: np.ndarray = None  # np.ndarray of shape (batch_size, 256, 1024), dtype=np.float32
    
    def add_result(self, vision_embedding, vision_second_last_layer_hidden_states, question_id):
        self.question_ids_list.append(question_id)
        self.vision_embedding_list.append(vision_embedding)
        self.vision_second_last_layer_hidden_states_list.append(vision_second_last_layer_hidden_states)
        self.num_queries += 1
        

    def serialize(self, start_pos, end_pos):
        """
        Serializes the following fields into a contiguous byte buffer:
          - Header: 4 bytes for batch_size (uint32).
          - question_id: fixed-size array of int64 (batch_size,).
          - vision_embedding: raw bytes of the array (batch_size, 32, 128), dtype np.float32.
          - vision_hidden_states: raw bytes of the array (batch_size, 256, 1024), dtype np.float32.
        """
        # Compute the total size
        batch_size = end_pos - start_pos
        qids_size = batch_size * np.dtype(np.int64).itemsize
        embedding_size = batch_size * 32 * 128 * np.dtype(np.float32).itemsize
        hidden_states_size = batch_size * 256 * 1024 * np.dtype(np.float32).itemsize
        total_size = 4 + qids_size + embedding_size + hidden_states_size
        
        # Allocate one contiguous buffer.
        serialized_buffer = np.zeros(total_size, dtype=np.uint8)
        # Determine segment positions
        qids_offset = 4
        embedding_offset = qids_offset + qids_size
        hidden_states_offset = embedding_offset + embedding_size
        # Get the array reference to write
        qid_array = np.frombuffer(serialized_buffer[qids_offset:qids_offset + qids_size],
                                    dtype=np.int64)
        embedding_array = np.frombuffer(serialized_buffer[embedding_offset:embedding_offset + embedding_size],
                                        dtype=np.float32).reshape((batch_size, 32, 128))
        hidden_states_array = np.frombuffer(serialized_buffer[hidden_states_offset:hidden_states_offset + hidden_states_size],
                                            dtype=np.float32).reshape((batch_size, 256, 1024))
        
        # Write Header: batch_size (as uint32)
        np.frombuffer(serialized_buffer[:4], dtype=np.uint32)[0] = batch_size
        # Write the data
        written_counter = 0  
        for manager_idx in range(start_pos, end_pos):
            qid_array[written_counter] = self.question_ids_list[manager_idx]
            embedding_array[written_counter] = self.vision_embedding_list[manager_idx]
            hidden_states_array[written_counter] = self.vision_second_last_layer_hidden_states_list[manager_idx]
            written_counter += 1
        return serialized_buffer
        

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous byte buffer back into the original fields.
        Expected layout (in order):
          - Header: 4 bytes (uint32: batch_size)
          - question_id: array of int64, shape (batch_size,)
          - vision_embedding: array of float32, shape (batch_size, 32, 128)
          - vision_hidden_states: array of float32, shape (batch_size, 256, 1024)
        """
        # # NOTE: need to create copy here to avoid buffer been overwriten, due to the aggregating mechanism required at this step
        # # fixed already
        # buffer = data.copy()
        offset = 0

        # --- Read header ---
        batch_size = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += 4
        
        qids_size = batch_size * np.dtype(np.int64).itemsize
        embedding_size = batch_size * 32 * 128 * np.dtype(np.float32).itemsize
        
        self.question_ids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset)
        offset += qids_size
        self.vision_embedding = np.frombuffer(buffer, dtype=np.float32, count=batch_size * 32 * 128, offset=offset).reshape((batch_size, 32, 128))
        offset += embedding_size
        self.vision_second_last_layer_hidden_states = np.frombuffer(buffer, dtype=np.float32, count=batch_size * 256 * 1024, offset=offset).reshape((batch_size, 256, 1024))

    def print_shape(self):
        print("--- StepBResultBatchManager shape info ---")
        print(f"question_ids: {self.question_ids.shape}")
        print(f"vision_embedding: {self.vision_embedding.shape}")
        print(f"vision_hidden_states: {self.vision_second_last_layer_hidden_states.shape}")

# ===============  Class and Serializer for Step D cross attention  ===============

class StepCDIntermediateResult:
    def __init__(self):
        self._question_id       = None
        self._np_text_sequence_bytes  = None
        self._input_ids         = None   # torch.Tensor on cpu, shape (1, 32)
        self._text_embeddings   = None   # torch.Tensor on cpu, shape (1, 32, 128)
        self._text_encoder_hidden_states = None   # torch.Tensor on cpu, shape (1, 32, 768)
        self._vision_embeddings = None   # torch.Tensor on cpu, shape (1, 32, 128)
        self._vision_second_last_layer_hidden_states = None    # torch.Tensor on cpu, shape (1, 256, 1024)
        
    def collected_all(self):
        has_all = self._np_text_sequence_bytes is not None and \
            self._input_ids is not None and \
            self._text_embeddings is not None and \
            self._text_encoder_hidden_states is not None and \
            self._vision_embeddings is not None and\
            self._vision_second_last_layer_hidden_states is not None and\
            self._question_id is not None
            
        return has_all

class PendingStepCDDataBatcher():
    '''
    Super batch of StepDMessageBatcher, used for model runtime batch execution
    '''
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        
        # Metadata, not used in computation, so could be in discontiguous memory
        self.question_ids = []     
        self.np_text_sequence_bytes = []
        
        # Inputs for step C and D
        # At initialization allocate memories on GPU
        self.input_ids = torch.empty((self.max_batch_size, 32), dtype=torch.long, device="cuda")
        self.text_embeddings = torch.empty((self.max_batch_size, 32, 128), dtype=torch.float32, device="cuda")
        self.text_encoder_hidden_states = torch.empty((self.max_batch_size, 32, 768), dtype=torch.float32, device="cuda")
        self.vision_embeddings = torch.empty((self.max_batch_size, 32, 128), dtype=torch.float32, device="cuda")        
        self.vision_second_last_layer_hidden_states = torch.empty((self.max_batch_size, 256, 1024), dtype=torch.float32, device="cuda")

    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, intermediate_result: StepCDIntermediateResult):
        self.question_ids.append(intermediate_result._question_id) # no copy here using reference
        self.np_text_sequence_bytes.append(intermediate_result._np_text_sequence_bytes) # no copy here using reference
        
        # Copy each tensor into the preallocated GPU tensors
        self.input_ids[self.num_pending].copy_(intermediate_result._input_ids.squeeze(0).to("cuda"))
        self.text_embeddings[self.num_pending].copy_(intermediate_result._text_embeddings.squeeze(0).to("cuda"))
        self.text_encoder_hidden_states[self.num_pending].copy_(intermediate_result._text_encoder_hidden_states.squeeze(0).to("cuda"))
        self.vision_embeddings[self.num_pending].copy_(intermediate_result._vision_embeddings.squeeze(0).to("cuda"))
        self.vision_second_last_layer_hidden_states[self.num_pending].copy_(intermediate_result._vision_second_last_layer_hidden_states.squeeze(0).to("cuda"))

        self.num_pending += 1
    
    def reset(self):
        '''
        Reset the fields
        '''
        self.question_ids = []
        self.np_text_sequence_bytes = []
        self.input_ids.fill_(0)
        self.text_embeddings.fill_(0)
        self.text_encoder_hidden_states.fill_(0)
        self.vision_embeddings.fill_(0)
        self.vision_second_last_layer_hidden_states.fill_(0)
        self.num_pending = 0


class StepDMessageBatcher:
    def __init__(self, max_batch_size = None):
        self.num_queries = 0
        self.question_ids = []       # List[int] of length batch_size.
        
        # Serialization fields
        self.max_batch_size = max_batch_size
        self.np_text_sequence_bytes = []  # List of reference to np.ndarray that is utf-8 encoded text_sequence, only decode it when asked
        self.query_embeddings_list = []   # List of reference np.ndarray of shape (320, 128), dtype=np.float32.
        self.text_pos = {}   # qid -> (offset, length)
        
        # Deserialization fields
        self.text_sequence = []     # List[str] of length batch_size.
        self.query_embeddings = None  # np.ndarray of shape (batch_size, 320, 128), dtype=np.float32.
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        
    def space_left(self):
        return self.max_batch_size - self.num_queries
    
    def add_results(self, question_id, text_sequence, query_embeddings, start_pos):
        num_to_add = min(self.space_left(), len(question_id) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(question_id[start_pos:end_pos])
        for i in range(start_pos, end_pos):
            # append the reference to the numpy arrays without copy
            self.np_text_sequence_bytes.append(text_sequence[i])
            self.query_embeddings_list.append(query_embeddings[i])
        self.num_queries += num_to_add
        return end_pos
    

    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous byte buffer:
          - Header: 4 bytes for batch_size (uint32).
          - Metadata for text_sequence: for each query, two int64 values:
                * query_offset: starting offset (in bytes) within the variable segment.
                * query_length: byte-length of the query.
          - Fixed segments:
                * question_ids: (batch_size,) int64.
                * query_embeddings: (batch_size, 320, 128) float32.
          - Variable segment:
                * text_sequence: concatenated UTF-8 encoded bytes.
        """
        # Compute the total size 
        batch_size = self.num_queries
        header_size = np.dtype(np.uint32).itemsize
        metadata_dtype = np.dtype([("query_offset", np.int64), ("query_length", np.int64)])
        metadata_size = batch_size * metadata_dtype.itemsize
        qids_size = batch_size * np.dtype(np.int64).itemsize
        query_embeddings_size = batch_size * 320 * 128 * np.dtype(np.float32).itemsize
        total_text_sequence_np_size = 0
        cur_text_offset = header_size + metadata_size + qids_size + query_embeddings_size
        for idx, q in enumerate(self.np_text_sequence_bytes):
            total_text_sequence_np_size += len(q)
            self.text_pos[self.question_ids[idx]] = (cur_text_offset, len(q))
            cur_text_offset += len(q)
        total_size = header_size + metadata_size + qids_size + query_embeddings_size + total_text_sequence_np_size
        
        # Allocate one contiguous buffer.
        serialized_buffer = np.empty(total_size, dtype=np.uint8)
        
        # Determine segment positions
        metadata_pos = header_size
        qids_offset = metadata_pos + metadata_size
        query_embeddings_offset = qids_offset + qids_size
        text_offset = query_embeddings_offset + query_embeddings_size

        # Get the array reference to write
        metadata_array = np.frombuffer(serialized_buffer[metadata_pos:metadata_pos + metadata_size], 
                                       dtype=metadata_dtype)
        qids_array = np.frombuffer(serialized_buffer[qids_offset:qids_offset + qids_size],
                                    dtype=np.int64)
        query_embeddings_array = np.frombuffer(serialized_buffer[query_embeddings_offset:query_embeddings_offset + query_embeddings_size],
                                                dtype=np.float32).reshape((batch_size, 320, 128))
        
        # Write Header: batch_size (as uint32)
        np.frombuffer(serialized_buffer[:header_size], dtype=np.uint32)[0] = batch_size
        written_counter = 0  # local position in this byte buffer
        for idx in range(batch_size):
            qid = self.question_ids[idx]
            abs_offset, qlen = self.text_pos[qid]
            metadata_array[written_counter]["query_offset"] = abs_offset
            metadata_array[written_counter]["query_length"] = qlen
            qids_array[written_counter] = qid
            query_embeddings_array[written_counter] = self.query_embeddings_list[idx]
            serialized_buffer[abs_offset:abs_offset + qlen] = self.np_text_sequence_bytes[idx]   # could use np.frombuffer(....)
            written_counter += 1
        return serialized_buffer

        
    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous byte buffer back into the original fields.
        Expected layout (in order):
          - Header: 4 bytes (uint32: batch_size)
          - Metadata for text_sequence: per query (query_offset and query_length as int64)
          - question_ids: array of int64, shape (batch_size,)
          - query_embeddings: array of float32, shape (batch_size, 320, 128)
          - text_sequence: concatenated UTF-8 bytes.
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header ---
        batch_size = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += 4
        
        metadata_dtype = np.dtype([("query_offset", np.int64), ("query_length", np.int64)])
        metadata_size = batch_size * metadata_dtype.itemsize
        qids_size = batch_size * np.dtype(np.int64).itemsize
        query_embeddings_size = batch_size * 320 * 128 * np.dtype(np.float32).itemsize
        
        metadata_array = np.frombuffer(buffer, dtype=metadata_dtype, count=batch_size, offset=offset)
        offset += metadata_size
        self.question_ids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset)
        offset += qids_size
        self.query_embeddings = np.frombuffer(buffer, dtype=np.float32, count=batch_size * 320 * 128, offset=offset).reshape((batch_size, 320, 128))
        offset += query_embeddings_size
        
        for m in metadata_array:
            start = int(m["query_offset"])
            length = int(m["query_length"])
            self.text_sequence.append(buffer[start:start+length].tobytes().decode("utf-8"))
        self.num_queries = batch_size
        
    def reset(self):
        '''
        Reset the fields
        '''
        self.question_ids = []
        self.np_text_sequence_bytes = []
        self.query_embeddings_list = []
        self.text_sequence = []
        self.num_queries = 0

    def print_shape(self):
        print("--- StepDMessageBatcher shape info ---")
        print(f"question_ids: {self.question_ids.shape}")
        print(f"query_embeddings: {self.query_embeddings.shape}")
        print(f"text_sequence: {len(self.text_sequence)}")


class PendingSearchBatcher():
    '''
    Super batch of Colbert Search, used for model runtime batch execution
    '''
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        # TODO: when using GPU RDMA direct, below fields should be allocated in CUDA memory
        self.question_ids = []      # List[int] of length batch_size.
        self.text_sequence = []     # List[str] of length batch_size.
        self.query_embeddings = torch.empty((self.max_batch_size, 320, 128), dtype=torch.float32, device="cuda")

    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, stepDMessageBatcher, start_pos):
        num_to_add = min(self.space_left(), stepDMessageBatcher.num_queries - start_pos)
        end_pos = start_pos + num_to_add
        # use a copy to avoid RMDA buffer being freed, since current implementation doesn't block the action queue at Cascade
        self.question_ids.extend(copy.deepcopy(stepDMessageBatcher.question_ids[start_pos:end_pos])) 
        self.text_sequence.extend(stepDMessageBatcher.text_sequence[start_pos:end_pos])
        self.query_embeddings[self.num_pending:self.num_pending + num_to_add].copy_(
            torch.tensor(stepDMessageBatcher.query_embeddings[start_pos:end_pos], device="cuda")
        )
        self.num_pending += num_to_add
        return end_pos

    def reset(self):
        '''
        Reset the fields
        '''
        self.question_ids = []
        self.text_sequence = []
        self.query_embeddings.fill_(0)
        self.num_pending = 0
        


    
if __name__ == "__main__":
    pass
    # test_list = []
    # b1 = PendingTextDataBatcher(100)
    # b1.question_ids = [1, 2, 3]
    # b1.text_sequence = ["hello", "world", "test"]
    # b1.input_ids = np.zeros((3, 32), dtype=np.int64)
    # b1.attention_mask = np.ones((3, 32), dtype=np.int64)
    # test_list.append(b1)
    # # print memory address of text_list[0]'s input_ids
    # print(f"add of text_list[0]: {test_list[0].input_ids.ctypes.data}")
    # print(f"addr of text_list : {id(test_list)}")
    # b2 = PendingTextDataBatcher(100)
    # b2.question_ids = [4, 5]
    # b2.text_sequence = ["hello", "world"]
    # b2.input_ids = np.zeros((2, 32), dtype=np.int64)
    # b2.attention_mask = np.ones((2, 32), dtype=np.int64)
    # test_list.append(b2)
    # # print memory address of text_list[0]'s input_ids
    # print(f"add of text_list[0]: {test_list[0].input_ids.ctypes.data}")
    # for i in range(100000):
    #     temp_b = PendingTextDataBatcher(100)
    #     temp_b.question_ids = [i]
    #     temp_b.text_sequence = ["hello"]
    #     temp_b.input_ids = np.zeros((1, 32), dtype=np.int64)
    #     temp_b.attention_mask = np.ones((1, 32), dtype=np.int64)
    #     test_list.append(temp_b)
    # print(f"add of text_list[0]: {test_list[0].input_ids.ctypes.data}")
    # print(f"addr of text_list : {id(test_list)}")

