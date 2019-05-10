from os import path
from vocabulary import Vocabulary
from config_loading import RawDataProcessingConfig, CorpusConfig

CORPUS_FILE = "cornell_movies.txt"

class Corpus:

	def __init__(self, data_filename):
		self._corpus_config = CorpusConfig()
		self._conversation_separator = RawDataProcessingConfig().conversation_separator
		self._data_filepath = path.join(self._corpus_config.corpus_folder, data_filename)

		self._file_lines = self._get_lines_from_file()
		self._dialogue_lines = [line for line in self._file_lines if line !=self._conversation_separator]
		self._dialogues = self._get_dialogue_lists()
		self._vocabulary = Vocabulary(self._dialogue_lines)
		self._seqs_data = self._build_seqs_pairs()


	def _get_lines_from_file(self):
		with open(self._data_filepath, 'r') as corpus_file:
			return [line.strip('\n') for line in corpus_file.readlines()]

	def _get_dialogue_lists(self):
		seperator_indices = [i for i, line in enumerate(self._file_lines) if line == self._conversation_separator]
		splits = [self._file_lines[:seperator_indices[0]]]
		for i in range(len(seperator_indices)-1):
			first = seperator_indices[i] + 1
			last = seperator_indices[i+1]
			splits.append(self._file_lines[first:last])
		splits.append(self._file_lines[seperator_indices[-1]+1:])
		return splits
	
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
	corpus = Corpus(CORPUS_FILE)
	data = corpus.seqs_data
	vocab = corpus.vocabulary
	#lengths = []
	#for pair in data:
	#	lengths.append(len(pair[0].split(" ")))
	#	lengths.append(len(pair[1].split(" ")))
	#print(list(sorted(lengths)))
	print(vocab.word_to_index("#UNK#"))
	