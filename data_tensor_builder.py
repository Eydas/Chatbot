import torch
from corpus_loader import Corpus, CORPUS_FILE

class BatchTensorBuilder:
    def __init__(self, seqs_pair_batch, vocabulary):
        self._vocabulary = vocabulary
        input_seqs, output_seqs = zip(*seqs_pair_batch)

        self._input_lengths = self._build_lengths_tensor(input_seqs)
        self._input_seqs_max_length = self._get_tensor_required_length(self._input_lengths)

        self._target_lengths = self._build_lengths_tensor(output_seqs)
        self._target_seqs_max_length = self._get_tensor_required_length(self._target_lengths)

        self._input_seqs_tensor = self._build_seqs_list_tensor(input_seqs, self._input_seqs_max_length)
        self._target_seqs_tensor = self._build_seqs_list_tensor(output_seqs, self._target_seqs_max_length)

        self._masks = self._build_seqs_mask_tensor(self._target_lengths, self._target_seqs_max_length)


    def _get_seqs_length(self, sequence):
        # Length of sequence includes the "End of Sequence token", hence the "+ 1"
        return len(sequence.split(' ')) + 1


    def _build_lengths_tensor(self, seqs_list):
        return torch.cat([torch.tensor([self._get_seqs_length(seq)]) for seq in seqs_list])


    def _get_tensor_required_length(self, lengths_tensor):
        return lengths_tensor.max().item()


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

    def _build_seq_mask_tensor(self, seq_length, required_seq_length):
        pad_length = required_seq_length - seq_length
        return torch.cat((torch.ones(seq_length), torch.zeros(pad_length)))

    def _build_seqs_mask_tensor(self, lengths_tensor, required_seq_length):
        return torch.stack([self._build_seq_mask_tensor(seq_length.item(), required_seq_length)
                            for seq_length in lengths_tensor])

    @property
    def input_lengths(self):
        return self._input_lengths


    @property
    def target_lengths(self):
        return self._target_lengths


    @property
    def input_seqs_tensor(self):
        return self._input_seqs_tensor.transpose(0, 1)


    @property
    def target_seqs_tensor(self):
        return self._target_seqs_tensor.transpose(0, 1)

    @property
    def masks(self):
        return self._masks.transpose(0, 1)


if __name__ == "__main__":
    corpus = Corpus(CORPUS_FILE)
    tensor_builder = BatchTensorBuilder(corpus.seqs_data[:5], corpus.vocabulary)
    print(tensor_builder.target_seqs_tensor)
    print(tensor_builder.masks)
    #print(tensor_builder.input_seqs_tensor)
    #print(tensor_builder.input_lengths)
    #print(tensor_builder.target_seqs_tensor)
    #print(tensor_builder.target_lengths)
