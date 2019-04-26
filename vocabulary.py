import math
from config_loading import VocabularyConfig

class Vocabulary:
	def __init__(self, lines):
		self._config = VocabularyConfig()

		self.words = {
			self._config.start_of_sequence_token: math.inf,
			self._config.end_of_sequence_token: math.inf,
			self._config.unknown_word_token: math.inf
		}
		self._trimmed = False

		for line in lines:
			self.add_line(line)
		if self._config.trim_vocabulary:
			self.trim_vocabulary()
		self.finalize()

	def add_word(self, word):
		if word not in self.words:
			self.words[word] = 1
		else:
			self.words[word] += 1
			
	def add_line(self, line):
		for word in line.strip('\n').strip().split(' '):
			self.add_word(word)
	
	def trim_vocabulary(self):
		if self._trimmed:
			return
		self._trimmed = True
		self.words = {word: occurences for word, occurences in self.words.items()
					  if occurences >= self._config.inclusion_threshold}
		
	def finalize(self):
		self._index_to_word = list(self.words.keys())
		self._word_to_index = {word: index for index, word in enumerate(self._index_to_word)}
		
	def index_to_word(self, index):
		return self._index_to_word[index]
		
	def word_to_index(self, word):
		return self._word_to_index[word]
		
	def get_words(self):
		return self._index_to_word

	@property
	def start_of_sequence_token_index(self):
		return self.word_to_index(self._config.start_of_sequence_token)

	@property
	def end_of_sequence_token_index(self):
		return self.word_to_index(self._config.end_of_sequence_token)

	@property
	def unknown_token_index(self):
		return self.word_to_index(self._config.unknown_word_token)
		