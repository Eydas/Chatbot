import corpus_vocabulary
from preprocess_lines import CONV_SEPERATOR

CORPUS_FOLDER = "./corpus/"
CORPUS_FILE = "movie_lines_processed.txt"

def get_lines_from_file(filename):
	with open(filename, 'r') as corpus:
		return [line.strip('\n') for line in corpus.readlines()]

def get_dialogue_lists(lines):
	seperator_indices = [i for i, line in enumerate(lines) if line == CONV_SEPERATOR.strip('\n')]
	splits = [lines[:seperator_indices[0]]]
	for i in range(len(seperator_indices)-1):
		first = seperator_indices[i] + 1
		last = seperator_indices[i+1]
		splits.append(lines[first:last])
	splits.append(lines[seperator_indices[-1]+1:])
	return splits
	
def build_seqs_pairs(dialogues):
	seqs_pairs = []
	for dialogue in dialogues:
		for i in range(0, len(dialogue)-1):
			seqs_pairs.append([dialogue[i], dialogue[i+1]])
	return seqs_pairs
	
def get_corpus_vocabulary(dialogues):
	vocabulary = corpus_vocabulary.corpus_vocabulary()
	for dialogue in dialogues:
		for line in dialogue:
			vocabulary.add_line(line)
	vocabulary.trim_vocabulary()
	vocabulary.finalize()
	return vocabulary
	
def common_words_only(line, vocabulary):
	words = line.split(' ')
	for word in words:
		if word not in vocabulary.words:
			return False
	return True
	
def filter_pairs_with_uncommon_words(pairs, vocabulary):
	return list(filter(lambda pair: common_words_only(pair[0], vocabulary) and common_words_only(pair[1], vocabulary) , pairs))

def load_seqs_data():
	corpus_file = CORPUS_FOLDER + CORPUS_FILE
	lines = get_lines_from_file(corpus_file)
	dialogues = get_dialogue_lists(lines)
	vocabulary = get_corpus_vocabulary(dialogues)
	seqs_pairs = build_seqs_pairs(dialogues)
	seqs_pairs_filtered = filter_pairs_with_uncommon_words(seqs_pairs, vocabulary)
	return seqs_pairs_filtered, vocabulary
	
if __name__ == "__main__":
	data, vocabulary = load_seqs_data()
	#lengths = []
	#for pair in data:
	#	lengths.append(len(pair[0].split(" ")))
	#	lengths.append(len(pair[1].split(" ")))
	print(len(data))
	