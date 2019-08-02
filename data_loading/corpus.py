import torch
from torch.utils.data import Dataset, DataLoader
from config_loading import CorpusConfig
from data_loading.vocabulary import Vocabulary
import json
import itertools

class CorpusDataset(Dataset):
    def __init__(self, corpus_file):
        super(CorpusDataset, self).__init__()
        self._config = CorpusConfig()
        self._corpus_filepath = corpus_file
        self._vocabulary = Vocabulary()

        # load data from corpus file and process it
        dialogues = self._get_dialogue_lists()
        all_lines = list(itertools.chain.from_iterable(dialogues))
        self._vocabulary.add_lines(all_lines)
        self._seqs_data = self._build_seqs_pairs(dialogues)
        self._seqs_data = self._filter_pairs_with_unknown_words_in_target()


    def __len__(self):
        return len(self._seqs_data)


    def __getitem__(self, idx):
        return self._seqs_data[idx]


    def build_batch(self, batch):
        input_seqs, output_seqs = zip(*batch)

        input_lengths = self._build_lengths_tensor(input_seqs)
        input_seqs_max_length = self._get_tensor_required_length(input_lengths)

        target_lengths = self._build_lengths_tensor(output_seqs)
        target_seqs_max_length = self._get_tensor_required_length(target_lengths)

        input_seqs_tensor = self._build_seqs_list_tensor(input_seqs, input_seqs_max_length)
        target_seqs_tensor = self._build_seqs_list_tensor(output_seqs, target_seqs_max_length)

        masks = self._build_seqs_mask_tensor(target_lengths, target_seqs_max_length)

        return input_seqs_tensor, input_lengths, target_seqs_tensor, masks


    def get_data_loader(self, batch_size):
        return DataLoader(self,
                          batch_size = batch_size,
                          shuffle = True,
                          collate_fn = self.build_batch,
                          pin_memory = True,
                          drop_last = False)


    def _get_tensor_required_length(self, lengths_tensor):
        return lengths_tensor.max().item()


    def _get_dialogue_lists(self):
        with open(self._corpus_filepath, 'r') as corpus_file:
            line_lists = json.load(corpus_file)
            line_lists = [[line.strip('\n').strip() for line in line_list]
                          for line_list in line_lists]
            return line_lists


    def _build_seqs_pairs(self, dialgoues):
        seqs_pairs = []
        for dialogue in dialgoues:
            for i in range(0, len(dialogue) - 1):
                seqs_pairs.append([dialogue[i], dialogue[i + 1]])
        return seqs_pairs


    def _known_words_only(self, line):
        return all(self._vocabulary.is_word_known(word) for word in line.split(' '))


    def _filter_pairs_with_unknown_words_in_target(self):
        return list(filter(lambda pair: self._known_words_only(pair[1]), self._seqs_data))


    def _get_seqs_length(self, sequence):
        # Length of sequence includes the "End of Sequence token", hence the "+ 1"
        return len(sequence.split(' ')) + 1


    def _build_lengths_tensor(self, seqs_list):
        return torch.cat([torch.tensor([self._get_seqs_length(seq)]) for seq in seqs_list])


    def _build_token_tensor(self, word):
        if self._vocabulary.is_word_known(word):
            return torch.tensor([self._vocabulary.word_to_index(word)])
        else:
            return torch.tensor([self._vocabulary.unknown_token_index])


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
        return torch.cat((torch.ones(seq_length).byte(), torch.zeros(pad_length).byte()))

    def _build_seqs_mask_tensor(self, lengths_tensor, required_seq_length):
        return torch.stack([self._build_seq_mask_tensor(seq_length.item(), required_seq_length)
                            for seq_length in lengths_tensor])


    @property
    def vocabulary(self):
        return self._vocabulary

    # Needed?
    @property
    def seqs_data(self):
        return self._seqs_data