from torch.utils.data import Dataset
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

    @property
    def vocabulary(self):
        return self._vocabulary

    # Needed?
    @property
    def seqs_data(self):
        return self._seqs_data