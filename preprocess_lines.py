from os import path
from config_loading import RawDataProcessingConfig
from text_normalization import TextNormalizer

RAW_DATA_FOLDER = "./data/raw/cornell_movie_dialogues"
RAW_DATA_FILENAME = "movie_lines.txt"

OUTPUT_CORPUS_FOLDER = "./data/corpus"
OUTPUT_CORPUS_FILENAME = "movie_lines_processed.txt"


LINE_MAX_ALLOWED_LENGTH = 20


class CornellCorpusPreProcessor:
    def __init__(self, raw_data_filepath):
        with open(raw_data_filepath, 'r') as raw_data_file:
            raw_data = raw_data_file.readlines()
        self._lines = [CornellLineData.from_line(raw_line) for raw_line in raw_data]
        text_normalizer = TextNormalizer()
        for line_obj in self._lines:
            line_obj.line = text_normalizer.normalize_text(line_obj.line)
        self._group_lines_into_conversations()

    def _group_lines_into_conversations(self):
        self._conversations = [[]]
        self._conversations[0].append(self._lines[0])
        for i in range(1, len(self._lines)):
            if self._lines[i].is_a_response_or_query_of(self._lines[i-1]):
                self._conversations[-1].append(self._lines[i])
            else:
                self._conversations.append([self._lines[i]])

        for conversation in self._conversations:
            conversation.sort(key = lambda line_data: line_data.line_id)


    @property
    def lines(self):
        return self._lines

    @property
    def conversations(self):
        return self._conversations


class CornellLineData:
    def __init__(self, line_id, character_id, movie_id, line):
        self.line_id = line_id
        self.character_id = character_id
        self.movie_id = movie_id
        self.line = line

    @staticmethod
    def from_line(line):
        fields = line.split(" +++$+++ ")
        return CornellLineData(int(fields[0][1:]), int(fields[1][1:]), int(fields[2][1:]), fields[4])

    def is_a_response_or_query_of(self, another_line):
        return self.movie_id == another_line.movie_id \
               and self.character_id != another_line.character_id \
               and (self.line_id == another_line.line_id + 1
                    or self.line_id == another_line.line_id - 1)

    def __repr__(self):
        return self.line


def process_lines(input_filepath, output_filepath):
    lines_data_obj = CornellCorpusPreProcessor(input_filepath)
    output_text = ("\n" + RawDataProcessingConfig().conversation_separator + "\n").join(
        ["\n".join([line_data.line for line_data in conversation])
         for conversation in lines_data_obj.conversations])

    with open(output_filepath, 'w') as outfile:
        outfile.write(output_text)

if __name__ == "__main__":
    process_lines(path.join(RAW_DATA_FOLDER, RAW_DATA_FILENAME), path.join(OUTPUT_CORPUS_FOLDER, OUTPUT_CORPUS_FILENAME))
