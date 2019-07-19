import json
from config_loading import VocabularyConfig

class Vocabulary:
	def __init__(self, lines = None):
		self._config = VocabularyConfig()

		if lines:
			self._words = {
				self._config.pad_token: self._create_token_listing(0, self._config.pad_token, None),
				self._config.start_of_sequence_token: self._create_token_listing(1, self._config.start_of_sequence_token, None),
				self._config.end_of_sequence_token: self._create_token_listing(2, self._config.end_of_sequence_token, None),
				self._config.unknown_word_token: self._create_token_listing(3, self._config.unknown_word_token, None)
			}

			self._trimmed = False

			for line in lines:
				self.add_line(line)
			if self._config.trim_vocabulary:
				self.trim_vocabulary()
			self.finalize()

	def add_word(self, word):
		if word not in self._words:
			self._words[word] = self._create_token_listing(len(self._words), word, 1)
		else:
			self._words[word]['occurences'] += 1
			
	def add_line(self, line):
		for word in line.strip('\n').strip().split(' '):
			self.add_word(word)
	
	def trim_vocabulary(self):
		if self._trimmed:
			return
		self._trimmed = True
		self._words = {word: data for word, data in self._words.items()
					   if data['occurences'] is None or data['occurences'] >= self._config.inclusion_threshold}
		
	def finalize(self):
		self._index_to_word = [data['token'] for data in list(sorted(self._words.values(), key=lambda x: x['index']))]
		self._word_to_index = {word: index for index, word in enumerate(self._index_to_word)}
		del self._words
		
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

		vocab_data = list(sorted([(data['token'], data['index']) for data in json_data], key=lambda x: x[1]))

		loaded_vocab = Vocabulary()

		loaded_vocab._word_to_index = {
			data[0]: data[1]
			for data in vocab_data
		}
		loaded_vocab._index_to_word = [data[0] for data in vocab_data]
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
		return self.word_to_index(self._config.unknown_word_token)

	@property
	def pad_token_index(self):
		return self.word_to_index(self._config.pad_token)