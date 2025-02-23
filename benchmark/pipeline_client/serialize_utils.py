import numpy as np
import sys


class DataBatcher:
    def __init__(self):
        # These will be set externally (or via an add_data method)
        # All lists must have the same length = batch_size.
        self.pixel_values = None         # A contiguous NumPy array of shape (batch_size, 1, 224, 224), dtype=np.float32
        self.text_sequence = []          # List of strings (one per query)
        self.question_ids = []           # List of ints (one per query)
        self.questions = []              # List of strings (one per query)
        
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
    
    def serialize(self) -> np.ndarray:
        """
        Serializes the batch into a contiguous NumPy byte array.
        Layout (in order):
          [batch_size (4 bytes)] |
          [metadata: one record per query] |
          [qids: int64 per query] |
          [questions: concatenated variable-length bytes] |
          [pixel_values: raw bytes of fixed-size tensor] |
          [text_sequence: concatenated variable-length bytes]
        The metadata structure (per query) stores:
          - question_offset, question_length (for the questions segment)
          - text_sequence_offset, text_sequence_length (for the text_sequence segment)
        Offsets are relative to the start of their respective segment.
        """
        batch_size = len(self.question_ids)
        if not (len(self.questions) == len(self.text_sequence) == batch_size):
            raise ValueError("All input lists must have the same length")
        if self.pixel_values is None or self.pixel_values.shape[0] != batch_size:
            raise ValueError("pixel_values must be provided and its first dimension equals batch_size")
        
        # === Prepare variable-length segments: questions and text_sequence ===
        question_encodings = [q.encode('utf-8') for q in self.questions]
        text_seq_encodings = [t.encode('utf-8') for t in self.text_sequence]
        
        # Compute cumulative offsets and total lengths for the questions segment.
        question_offsets = []
        offset = 0
        for b in question_encodings:
            question_offsets.append(offset)
            offset += len(b)
        total_questions_size = offset
        
        # Compute cumulative offsets and total lengths for the text_sequence segment.
        text_seq_offsets = []
        offset = 0
        for b in text_seq_encodings:
            text_seq_offsets.append(offset)
            offset += len(b)
        total_text_seq_size = offset
        
        # === Compute sizes for fixed-length segments ===
        # Header: 4 bytes for batch_size.
        header_size = 4
        
        # Metadata: one record per query.
        metadata_dtype = np.dtype([
            ('question_offset', np.uint32),
            ('question_length', np.uint32),
            ('text_sequence_offset', np.uint32),
            ('text_sequence_length', np.uint32),
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        
        # QIDs: one int64 per query.
        qids_size = batch_size * np.dtype(np.int64).itemsize
        
        # Pixel values: assumed to be a contiguous array of shape (batch_size, 1, 224, 224)
        pixel_values_size = self.pixel_values.nbytes  # For example: batch_size * 1 * 224 * 224 * 4
        
        # Total size of the final buffer:
        total_size = (header_size + metadata_size +
                      qids_size +
                      total_questions_size +
                      pixel_values_size +
                      total_text_seq_size)
        
        # === Allocate buffer once ===
        buffer = np.zeros(total_size, dtype=np.uint8)
        
        # --- Write header: batch_size (uint32) ---
        np.frombuffer(buffer[:4], dtype=np.uint32)[0] = batch_size
        
        # --- Write metadata ---
        metadata_start = header_size
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size],
                                       dtype=metadata_dtype)
        for i in range(batch_size):
            metadata_array[i]['question_offset'] = question_offsets[i]
            metadata_array[i]['question_length'] = len(question_encodings[i])
            metadata_array[i]['text_sequence_offset'] = text_seq_offsets[i]
            metadata_array[i]['text_sequence_length'] = len(text_seq_encodings[i])
        
        # --- Write qids ---
        qids_start = metadata_start + metadata_size
        qids_array = np.frombuffer(buffer[qids_start:qids_start+qids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        
        # --- Write questions segment ---
        questions_start = qids_start + qids_size
        pos = questions_start
        for b in question_encodings:
            n = len(b)
            buffer[pos:pos+n] = np.frombuffer(b, dtype=np.uint8)
            pos += n
        
        # --- Write pixel_values segment ---
        pixel_values_start = questions_start + total_questions_size
        pixel_bytes = self.pixel_values.tobytes()  # Already contiguous
        buffer[pixel_values_start:pixel_values_start+pixel_values_size] = np.frombuffer(pixel_bytes, dtype=np.uint8)
        
        # --- Write text_sequence segment ---
        text_seq_start = pixel_values_start + pixel_values_size
        pos = text_seq_start
        for b in text_seq_encodings:
            n = len(b)
            buffer[pos:pos+n] = np.frombuffer(b, dtype=np.uint8)
            pos += n
        
        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous NumPy byte array back into its original fields.
        (It assumes the same layout as produced by `serialize`.)
        """
        self._bytes = data
        buffer = data.tobytes()  # Work with the raw bytes
        offset = 0
        
        # --- Read header: batch_size ---
        batch_size = np.frombuffer(buffer[offset:offset+4], dtype=np.uint32)[0]
        offset += 4
        
        # --- Read metadata ---
        metadata_dtype = np.dtype([
            ('question_offset', np.uint32),
            ('question_length', np.uint32),
            ('text_sequence_offset', np.uint32),
            ('text_sequence_length', np.uint32),
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer[offset:offset+metadata_size], dtype=metadata_dtype)
        offset += metadata_size
        
        # --- Read qids ---
        qids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer[offset:offset+qids_size], dtype=np.int64).tolist()
        offset += qids_size
        
        # --- Read questions segment ---
        total_questions_size = sum(metadata_array['question_length'])
        questions_bytes = buffer[offset:offset+total_questions_size]
        offset += total_questions_size
        questions = []
        for m in metadata_array:
            start = m['question_offset']
            length = m['question_length']
            q = questions_bytes[start:start+length].decode('utf-8')
            questions.append(q)
        
        # --- Read pixel_values segment ---
        # We assume the shape is known: (batch_size, 1, 224, 224) with dtype float32.
        pixel_values_size = batch_size * 1 * 224 * 224 * np.dtype(np.float32).itemsize
        pixel_values_bytes = buffer[offset:offset+pixel_values_size]
        offset += pixel_values_size
        pixel_values = np.frombuffer(pixel_values_bytes, dtype=np.float32).reshape((batch_size, 1, 224, 224))
        
        # --- Read text_sequence segment ---
        total_text_seq_size = sum(metadata_array['text_sequence_length'])
        text_seq_bytes = buffer[offset:offset+total_text_seq_size]
        offset += total_text_seq_size
        text_sequence = []
        for m in metadata_array:
            start = m['text_sequence_offset']
            length = m['text_sequence_length']
            t = text_seq_bytes[start:start+length].decode('utf-8')
            text_sequence.append(t)
        
        # Restore fields
        self.question_ids = qids
        self.questions = questions
        self.pixel_values = pixel_values
        self.text_sequence = text_sequence

    def get_data(self):
        """Returns a dictionary with all the batch data."""
        return {
            "question_ids": self.question_ids,
            "questions": self.questions,
            "pixel_values": self.pixel_values,
            "text_sequence": self.text_sequence
        }

# ---------------------------
# Example usage:
# ---------------------------
if __name__ == '__main__':
    # Suppose we have a batch of 3 queries.
    batcher = DataBatcher()
    batcher.question_ids = [101, 102, 103]
    batcher.questions = ["What is AI?", "How to serialize data?", "Example question."]
    batcher.text_sequence = ["AI", "serialize", "example"]
    # Create dummy pixel values (3 images of shape (1, 224, 224) with float32)
    batcher.pixel_values = np.random.rand(3, 1, 224, 224).astype(np.float32)
    
    # Serialize into a single byte buffer.
    serialized = batcher.serialize()
    print(type(serialized))
    # To test deserialization:
    new_batcher = DataBatcher()
    new_batcher.deserialize(serialized)
    
    print(f"Got message size of {sys.getsizeof(serialized)}")
    # Verify round-trip equality.
    data = new_batcher.get_data()
    print("Deserialized question_ids:", data["question_ids"])
    print("Deserialized questions:", data["questions"])
    print("Deserialized text_sequence:", data["text_sequence"])
    print("Deserialized pixel_values shape:", data["pixel_values"].shape)
