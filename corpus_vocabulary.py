class corpus_vocabulary:
	def __init__(self):
		self.words = {}
		self.trimmed = False
		self.trim_threshold = 3
		
	def add_word(self, word):
		if word not in self.words:
			self.words[word] = 1
		else:
			self.words[word] += 1
			
	def add_line(self, line):
		for word in line.split(' '):
			self.add_word(word)
	
	def trim_vocabulary(self):
		if self.trimmed:
			return
		self.trimmed = True
		self.words = {word: occurences for word, occurences in self.words.items() if occurences >= self.trim_threshold}
		
	def finalize(self):
		self._index_to_word = list(self.words.keys())
		self._word_to_index = {word: index for index, word in enumerate(self._index_to_word)}
		
	def index_to_word(self, index):
		return self._index_to_word[index]
		
	def word_to_index(self, word):
		return self._word_to_index[word]
		
	def get_words(self):
		return self._index_to_word
		