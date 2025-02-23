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
            
        print(f"Got questios: {questions}")
        # --- Read text_sequence segment ---
        total_text_seq_size = sum(int(x) for x in metadata_array["text_sequence_length"])
        text_seq_bytes = buffer[offset:offset+total_text_seq_size]
        offset += total_text_seq_size
        text_sequence = []
        for m in metadata_array:
            start = int(m["text_sequence_offset"])
            length = int(m["text_sequence_length"])
            text_sequence.append(text_seq_bytes[start:start+length].tobytes().decode("utf-8"))
            
        print(f"Got text seq: {text_sequence}")
        # --- Read pixel_values segment ---
        pixel_values_size = batch_size * 1 * 224 * 224 * np.dtype(np.float32).itemsize
        pixel_values_bytes = buffer[offset:offset+pixel_values_size]
        offset += pixel_values_size
        pixel_values = np.frombuffer(pixel_values_bytes, dtype=np.float32).reshape((batch_size, 1, 224, 224))
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