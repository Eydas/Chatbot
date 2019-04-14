import numpy as np
import unicodedata
import re

LINES_FILE_FOLDER = "./corpus/"
LINES_FILE_NAME = "movie_lines.txt"

OUTPUT_CORPUS_FILE = "movie_lines_processed.txt"

# replace x96 with '-'

CONV_SEPERATOR = "########\n"
LINE_MAX_ALLOWED_LENGTH = 20

CHARACTER_REPLACEMENTS = [
	('<i>', ''), 
	('</i>', ''),
	('<u>', ''), 
	('</u>', ''),
	('<b>', ''), 
	('</b>', ''),
	('[0]', ''),
	('[1]', ''),
	('[2]', ''),
	('[3]', ''),
	('[4]', ''),
	('[5]', ''),
	('[6]', ''),
	('[7]', ''),
	('[8]', ''),
	('[9]', '')
	]

def extract_fields_from_data_line(line_text):
	fields = line_text.split(" +++$+++ ")
	return [int(fields[0][1:]), int(fields[1][1:]), int(fields[2][1:]), normalize_line(fields[4]) + '\n']
	
def load_data_from_file(lines_file_path):
	line_data = []
	with open(lines_file_path) as lines_file:
		line_data = [extract_fields_from_data_line(line_text) for line_text in lines_file.readlines()]
	return line_data
	
def replace_bad_chars(text_line, char_replacements):
	for replacement in char_replacements:
		text_line = text_line.replace(replacement[0], replacement[1])
	return text_line
	
def unicode_to_ascii(line):
	return ''.join(ch for ch in unicodedata.normalize('NFD', line) if unicodedata.category(ch) != 'Mn')
	
def normalize_line(line):
	if line == CONV_SEPERATOR:
		return line
	normalized_line = unicode_to_ascii(replace_bad_chars(line.lower().strip(), CHARACTER_REPLACEMENTS))
	normalized_line = re.sub(r'([.?!])', r' \1', normalized_line)
	normalized_line = re.sub(r'[^a-zA-Z.?!]', r' ', normalized_line)
	normalized_line = re.sub(r'\s+', r' ', normalized_line)
	normalized_line = normalized_line.strip()
	return normalized_line
	
def is_allowed_length(line):
	return len(line.split(" ")) <= LINE_MAX_ALLOWED_LENGTH
	
def split_according_to_movie_field(data):
	split_lists = {}
	for datapoint in data:
		if datapoint[2] not in split_lists:
			split_lists[datapoint[2]] = []
		split_lists[datapoint[2]].append([datapoint[0], datapoint[1], datapoint[3]])
	return list(split_lists.values())

def sort_line_lists(line_lists):
	for line_list in line_lists:
		line_list.sort(key = lambda datapoint: (datapoint[0]))
		
def create_dialogue_lists(line_list):
	dialogue_list = []
	in_conv = False
	for i in range(0, len(line_list) - 1):
		datapoint1 = line_list[i]
		datapoint2 = line_list[i+1]
		if datapoint2[0] - datapoint1[0] == 1 and datapoint1[1] != datapoint2[1] and is_allowed_length(datapoint1[2]) and is_allowed_length(datapoint2[2]):
			if not in_conv:
				dialogue_list.append(datapoint1[2])
				in_conv = True
			dialogue_list.append(datapoint2[2])
		else:
			if in_conv:
				dialogue_list.append(CONV_SEPERATOR)
				in_conv = False
	return dialogue_list
	
def get_dialogues(split_line_lists):
	dialogue_list = []
	for line_list in split_line_lists:
		dialogue_list += create_dialogue_lists(line_list)
		if dialogue_list[-1] != CONV_SEPERATOR:
			dialogue_list.append(CONV_SEPERATOR)
	del dialogue_list[-1]	# remove final CONV_SEPERATOR
	return dialogue_list
	
def process_lines():
	data = load_data_from_file(LINES_FILE_FOLDER + LINES_FILE_NAME)
	split_lists = split_according_to_movie_field(data)
	sort_line_lists(split_lists)
	dialogue_list = get_dialogues(split_lists)
	
	output_file = LINES_FILE_FOLDER + OUTPUT_CORPUS_FILE
	with open(output_file, 'w') as outfile:
		outfile.writelines(dialogue_list)

if __name__ == "__main__":
	process_lines()
	