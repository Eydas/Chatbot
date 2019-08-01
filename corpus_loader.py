from os import path
from vocabulary import Vocabulary
from config_loading import CorpusConfig
import json
import itertools

CORPUS_FILE = "cornell_movies.json"

class Corpus:

	def __init__(self, data_filename):
		self._corpus_config = CorpusConfig()
		self._data_filepath = path.join(self._corpus_config.corpus_folder, data_filename)

		self._dialogues = self._get_dialogue_lists()
		self._all_lines = list(itertools.chain.from_iterable(self._dialogues))

		self._vocabulary = Vocabulary()
		self._vocabulary.add_lines(self._all_lines)
		self._seqs_data = self._build_seqs_pairs()
		self._seqs_data = self._filter_pairs_with_unknown_words_in_target()


	def _get_all_lines_from_file(self):
		with open(self._data_filepath, 'r') as corpus_file:
			return [line.strip('\n').strip() for line in corpus_file.readlines()]


	def _get_dialogue_lists(self):
		with open(self._data_filepath, 'r') as corpus_file:
			line_lists = json.load(corpus_file)
			line_lists = [[line.strip('\n').strip() for line in line_list]
					for line_list in line_lists]
			return line_lists


	def _build_seqs_pairs(self):
		seqs_pairs = []
		for dialogue in self._dialogues:
			for i in range(0, len(dialogue)-1):
				seqs_pairs.append([dialogue[i], dialogue[i+1]])
		return seqs_pairs


	def _known_words_only(self, line):
		return all(self._vocabulary.is_word_known(word) for word in line.split(' '))


	def _filter_pairs_with_unknown_words_in_target(self):
		return list(filter(lambda pair: self._known_words_only(pair[1]), self._seqs_data))


	@property
	def vocabulary(self):
		return self._vocabulary


	@property
	def seqs_data(self):
		return self._seqs_data

	
if __name__ == "__main__":
	corpus = Corpus(CORPUS_FILE)
	#data = corpus.seqs_data
	#vocab = corpus.vocabulary
	#vocab.save('vocab_save_test.json')
	#vocab2 = Vocabulary.load('vocab_save_test.json')
	print(len(corpus.seqs_data))
	#lengths = []
	#for pair in data:
	#	lengths.append(len(pair[0].split(" ")))
	#	lengths.append(len(pair[1].split(" ")))
	#print(list(sorted(lengths)))
	#print(vocab.word_to_index("#UNK#"))
	