from abc import ABC, abstractmethod
from typing import List, Text


STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']


class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass

    def get_character_start_stop_indices(self, input_data_offset, answer):
        """Remove any output following (and including) a stop_sequence.

        Some outputs start with newlines (unfortunately). We strip these, in
        order to ensure generations with greater-than-zero length.
        """
        start_index = input_data_offset

        # Strip zero-length generations from beginning and add to `input_data_offset`.
        newline = '\n'
        while answer[start_index:].startswith(newline):
            start_index += len(newline)

        # Get character index of first stop sequence
        stop_index = len(answer)
        for word in self.stop_sequences:
            index = answer[start_index:].find(word)
            if index != -1 and index + start_index < stop_index:
                stop_index = index + start_index

        return start_index, stop_index
