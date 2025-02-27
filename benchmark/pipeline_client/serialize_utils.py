import numpy as np

class DataBatcher:
    def __init__(self):
        # All fields must have the same batch size.
        self.pixel_values = None         # np.ndarray of shape (batch_size, 1, 224, 224) and dtype=np.float32
        self.text_sequence = []          # List of strings (one per query)
        self.question_ids = []           # List of ints (one per query)
        self.questions = []              # List of strings (one per query)
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
    def utf8_length(self, s: str) -> int:
        return len(s.encode("utf-8"))  # More efficient way to compute UTF-8 byte length
    
    def serialize(self) -> np.ndarray:

        batch_size = len(self.question_ids)
        if not (len(self.questions) == len(self.text_sequence) == batch_size):
            raise ValueError("All input lists must have the same length")
        if self.pixel_values is None or self.pixel_values.shape[0] != batch_size:
            raise ValueError("pixel_values must be provided and its first dimension must equal batch_size")
        # Prepare variable-length segments: questions and text_sequence.
        question_encodings = [q.encode("utf-8") for q in self.questions]
        text_seq_encodings = [t.encode("utf-8") for t in self.text_sequence]
        # Compute offsets for questions segment.
        question_offsets = []
        offset = 0
        for enc in question_encodings:
            question_offsets.append(offset)
            offset += len(enc)
        total_questions_size = offset
        # Compute offsets for text_sequence segment.
        text_seq_offsets = []
        offset = 0
        for enc in text_seq_encodings:
            text_seq_offsets.append(offset)
            offset += len(enc)
        total_text_seq_size = offset
        # Fixed-length parts.
        header_size = 4  # 4 bytes for batch_size (uint32).
        metadata_dtype = np.dtype([
            ("question_offset", np.int64),
            ("question_length", np.int64),
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64),
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        qids_size = batch_size * np.dtype(np.int64).itemsize
        pixel_values_size = self.pixel_values.nbytes  # e.g., batch_size * 1 * 224 * 224 * 4
        total_size = (header_size + metadata_size + qids_size +
                      total_questions_size + total_text_seq_size + pixel_values_size)
        # Allocate one contiguous buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        # --- Write header ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        # --- Write metadata ---
        metadata_start = header_size
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size],
                                       dtype=metadata_dtype)
        for i in range(batch_size):
            metadata_array[i]["question_offset"] = question_offsets[i]
            metadata_array[i]["question_length"] = len(question_encodings[i])
            metadata_array[i]["text_sequence_offset"] = text_seq_offsets[i]
            metadata_array[i]["text_sequence_length"] = len(text_seq_encodings[i])
        # --- Write question_ids ---
        qids_start = metadata_start + metadata_size
        qids_array = np.frombuffer(buffer[qids_start:qids_start+qids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        # --- Write questions segment ---
        questions_start = qids_start + qids_size
        pos = questions_start
        for enc in question_encodings:
            n = len(enc)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        # --- Write text_sequence segment ---
        text_seq_start = questions_start + total_questions_size
        pos = text_seq_start
        for enc in text_seq_encodings:
            n = len(enc)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        # --- Write pixel_values segment ---
        pixel_values_start = text_seq_start + total_text_seq_size
        pixel_bytes = self.pixel_values.tobytes()  # Get contiguous raw bytes.
        buffer[pixel_values_start:pixel_values_start+pixel_values_size] = np.frombuffer(pixel_bytes, dtype=np.uint8)
        self._bytes = buffer
        return buffer
    
    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous byte array back into the original fields.
        Assumes the same layout as produced by serialize().
        """
        self._bytes = data
        buffer = data
        offset = 0
        # --- Read header ---
        batch_size = int(np.frombuffer(buffer[offset:offset+4], dtype=np.uint32)[0])
        offset += 4
        # --- Read metadata ---
        metadata_dtype = np.dtype([
            ("question_offset", np.int64),
            ("question_length", np.int64),
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64),
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer[offset:offset+metadata_size], dtype=metadata_dtype)
        offset += metadata_size
        # --- Read question_ids ---
        qids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer[offset:offset+qids_size], dtype=np.int64).tolist()
        offset += qids_size
        # --- Read questions segment ---
        total_questions_size = sum(int(x) for x in metadata_array["question_length"])
        questions_bytes = buffer[offset:offset+total_questions_size]
        offset += total_questions_size
        questions = []
        for m in metadata_array:
            start = int(m["question_offset"])
            length = int(m["question_length"])
            questions.append(questions_bytes[start:start+length].tobytes().decode("utf-8"))
            
        # --- Read text_sequence segment ---
        total_text_seq_size = sum(int(x) for x in metadata_array["text_sequence_length"])
        text_seq_bytes = buffer[offset:offset+total_text_seq_size]
        offset += total_text_seq_size
        text_sequence = []
        for m in metadata_array:
            start = int(m["text_sequence_offset"])
            length = int(m["text_sequence_length"])
            text_sequence.append(text_seq_bytes[start:start+length].tobytes().decode("utf-8"))
            
        # --- Read pixel_values segment ---
        pixel_values_size = batch_size * 1 * 3 * 224 * 224 * np.dtype(np.float32).itemsize
        pixel_values_bytes = buffer[offset:offset+pixel_values_size]
        offset += pixel_values_size
        pixel_values = np.frombuffer(pixel_values_bytes, dtype=np.float32).reshape((batch_size, 1, 3,224, 224))
        # Restore fields.
        
        self.question_ids = qids
        self.questions = questions
        self.pixel_values = pixel_values
        self.text_sequence = text_sequence
        
    def get_data(self):
        return {
            "question_ids": self.question_ids,
            "questions": self.questions,
            "pixel_values": self.pixel_values,
            "text_sequence": self.text_sequence
        }
        
        
class PixelValueBatcher:
    def __init__(self):
        # Expected shape: (batch_size, 1, 3, 224, 224) and dtype=np.float32.
        self.pixel_values: np.ndarray = None
        # Query IDs as a NumPy array of shape (batch_size,) and dtype=np.int64.
        self.question_ids: np.ndarray = None
        # Internal buffer (serialized data)
        self._bytes: np.ndarray = None

    def serialize(self) -> np.ndarray:
        """
        Serialize the pixel_values and question_ids into a contiguous buffer.
        Layout:
        [4 bytes: batch_size (uint32)] | [question_ids (int64)] | [pixel_values (float32)]
        """
        if self.pixel_values is None or self.question_ids is None:
            raise ValueError("Both pixel_values and question_ids must be provided.")
        
        # Ensure question_ids is a NumPy array.
        question_ids_array = np.asarray(self.question_ids, dtype=np.int64)
        batch_size = question_ids_array.shape[0]
        
        if self.pixel_values.shape[0] != batch_size:
            raise ValueError("The first dimension of pixel_values must match the number of question_ids.")
        
        # Calculate sizes.
        header_size = np.dtype(np.uint32).itemsize  # 4 bytes
        question_ids_size = question_ids_array.nbytes       # batch_size * 8 bytes
        pixel_values_size = self.pixel_values.nbytes  # batch_size * 1 * 3 * 224 * 224 * 4
        
        total_size = header_size + question_ids_size + pixel_values_size
        # Allocate one contiguous buffer.
        buffer = np.empty(total_size, dtype=np.uint8)
        offset = 0

        # --- Write header: batch_size ---
        header_arr = np.array([batch_size], dtype=np.uint32)
        # Create a view into header_arr as uint8 (zero-copy view) and copy into the buffer slice.
        buffer[offset:offset + header_size] = header_arr.view(np.uint8)
        offset += header_size

        # --- Write question_ids ---
        # Use a view of the question_ids array as uint8.
        buffer[offset:offset + question_ids_size] = question_ids_array.view(np.uint8)
        offset += question_ids_size

        # --- Write pixel_values ---
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

        # --- Read question_ids ---
        question_ids_count = batch_size  # one int64 per example
        question_ids = np.frombuffer(buffer, dtype=np.int64, count=question_ids_count, offset=offset)
        offset += question_ids.nbytes  # or batch_size * np.dtype(np.int64).itemsize

        # --- Read pixel_values ---
        pixel_shape = (batch_size, 1, 3, 224, 224)
        num_pixels = np.prod(pixel_shape)
        # Create a zero-copy view and then reshape it
        pixel_values = np.frombuffer(buffer, dtype=np.float32, count=num_pixels, offset=offset).reshape(pixel_shape)
        offset += num_pixels * np.dtype(np.float32).itemsize

        # Assign fields (these are zero-copy views into the original buffer)
        self.question_ids = question_ids
        self.pixel_values = pixel_values

    def get_data(self):
        return {
            "question_ids": self.question_ids,
            "pixel_values": self.pixel_values
        }
