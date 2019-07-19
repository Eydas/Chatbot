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

		self._vocabulary = Vocabulary(self._all_lines)
		self._seqs_data = self._build_seqs_pairs()

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

	"""
	def common_words_only(line, vocabulary):
		words = line.split(' ')
		for word in words:
			if word not in vocabulary.words:
				return False
		return True

	def filter_pairs_with_uncommon_words(pairs, vocabulary):
		return list(filter(lambda pair: common_words_only(pair[0], vocabulary) and common_words_only(pair[1], vocabulary) , pairs))
	"""

	@property
	def vocabulary(self):
		return self._vocabulary

	@property
	def seqs_data(self):
		return self._seqs_data

	
if __name__ == "__main__":
	#corpus = Corpus(CORPUS_FILE)
	#data = corpus.seqs_data
	#vocab = corpus.vocabulary
	vocab = Vocabulary.load('vocab_save_test.json')
	print(vocab._index_to_word)
	#lengths = []
	#for pair in data:
	#	lengths.append(len(pair[0].split(" ")))
	#	lengths.append(len(pair[1].split(" ")))
	#print(list(sorted(lengths)))
	#print(vocab.word_to_index("#UNK#"))
	