import json
from config_loading import VocabularyConfig

class Vocabulary:
	def __init__(self):
		self._config = VocabularyConfig()

		self._words = {
			self._config.pad_token: self._create_token_listing(self._config.pad_index, self._config.pad_token, None),
			self._config.start_of_sequence_token: self._create_token_listing(self._config.start_of_sequence_index,
																			 self._config.start_of_sequence_token,
																			 None),
			self._config.end_of_sequence_token: self._create_token_listing(self._config.end_of_sequence_index,
																		   self._config.end_of_sequence_token,
																		   None),
			self._config.unknown_token: self._create_token_listing(self._config.unknown_index,
																		self._config.unknown_token,
																		None)
		}
		self._finalize()


	def add_lines(self, lines):
		self._trimmed = False

		for line in lines:
			self._add_line(line)
		if self._config.trim_vocabulary:
			self._trim_vocabulary()
		self._finalize()


	def _add_word(self, word):
		if word not in self._words:
			self._words[word] = self._create_token_listing(len(self._words), word, 1)
		else:
			self._words[word]['occurences'] += 1


	def _add_line(self, line):
		for word in line.strip('\n').strip().split(' '):
			self._add_word(word)


	def _trim_vocabulary(self):
		if self._trimmed:
			return
		self._trimmed = True
		self._words = {word: data for word, data in self._words.items()
					   if data['occurences'] is None or data['occurences'] >= self._config.inclusion_threshold}


	def _finalize(self):
		self._index_to_word = [data['token'] for data in list(sorted(self._words.values(), key=lambda x: x['index']))]
		self._word_to_index = {word: index for index, word in enumerate(self._index_to_word)}


	def is_word_known(self, word):
		return word in self._words


	def index_to_word(self, index):
		return self._index_to_word[index]


	def word_to_index(self, word):
		return self._word_to_index[word]


	def get_words(self):
		return self._index_to_word


	def save(self, file_path):
		json_data = [{
			'token': token,
			'index': index
		} for token, index in self._word_to_index.items()]

		with open(file_path, 'w') as jfp:
			json.dump(json_data, jfp)


	@staticmethod
	def load(file_path):
		# TODO: ensure correctness of data in file
		with open(file_path, 'r') as jfp:
			json_data = json.load(jfp)

		loaded_vocab = Vocabulary()

		loaded_vocab._words = {
			data['token']: data for data in json_data
		}
		loaded_vocab._finalize()
		loaded_vocab._trimmed = True

		return loaded_vocab

	def _create_token_listing(self, idx, token, init_occurences = None):
		return {
			'index': idx,
			'token': token,
			'occurences': init_occurences
		}


	@property
	def size(self):
		return len(self._index_to_word)


	@property
	def start_of_sequence_token_index(self):
		return self.word_to_index(self._config.start_of_sequence_token)


	@property
	def end_of_sequence_token_index(self):
		return self.word_to_index(self._config.end_of_sequence_token)


	@property
	def unknown_token_index(self):
		return self.word_to_index(self._config.unknown_token)


	@property
	def pad_token_index(self):
		return self.word_to_index(self._config.pad_token)