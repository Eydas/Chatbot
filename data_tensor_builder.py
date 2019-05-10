import torch
from corpus_loader import Corpus, CORPUS_FILE

class DataTensorBuilder:
    def __init__(self, corpus):
        self._corpus = corpus
        self._vocabulary = corpus.vocabulary
        input_seqs, output_seqs = zip(*corpus.seqs_data)
        self._input_seqs_max_length = max([len(input_seq.split(' ')) for input_seq in input_seqs])
        self._output_seqs_max_length = max([len(output_seq.split(' ')) for output_seq in output_seqs])

        self._input_seqs_tensor = self._build_seqs_list_tensor(input_seqs, self._input_seqs_max_length)
        self._output_seqs_tensor = self._build_seqs_list_tensor(output_seqs, self._output_seqs_max_length)



    def _build_token_tensor(self, word):
        if word not in self._vocabulary.words:
            return torch.tensor([self._vocabulary.unknown_token_index])
        else:
            return torch.tensor([self._vocabulary.word_to_index(word)])

    def _zero_pad_sequence(self, sequence_tensor_list, required_seq_length):
        pad_length = required_seq_length - len(sequence_tensor_list)
        return sequence_tensor_list + pad_length * [torch.tensor([self._vocabulary.pad_token_index])]


    def _build_sequence_tensor(self, sequence, required_seq_length):
        tokens = sequence.strip('\n').strip().split(' ')
        tensor_list = self._zero_pad_sequence([self._build_token_tensor(token) for token in tokens]
                         + [torch.tensor([self._vocabulary.end_of_sequence_token_index])], required_seq_length)
        return torch.cat(tensor_list)

    def _build_seqs_list_tensor(self, seqs_list, required_seq_length):
        return torch.stack([self._build_sequence_tensor(seq, required_seq_length) for seq in seqs_list])

    @property
    def input_seqs_tensor(self):
        return self._input_seqs_tensor

    @property
    def output_seqs_tensor(self):
        return self._output_seqs_tensor


if __name__ == "__main__":
    corpus = Corpus(CORPUS_FILE)
    tensor_builder = DataTensorBuilder(corpus)
    print(tensor_builder.input_seqs_tensor)
    #print(embedder.embed_sentence(data[0][0]))
