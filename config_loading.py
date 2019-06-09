import json
from os import path

class Config:

    def get_property(self, property_name):
        return self._config_dict[property_name]

class JsonConfig(Config):
    CONFIG_FOLDER = './config'

    def __init__(self, filename):
        with open(path.join(JsonConfig.CONFIG_FOLDER, filename), 'r') as json_file:
            self._config_dict = json.load(json_file)


class TextNormalizationConfig(JsonConfig):
    CONFIG_FILENAME = 'text_normalization.json'

    def __init__(self):
        super(TextNormalizationConfig, self).__init__(TextNormalizationConfig.CONFIG_FILENAME)
        self._config_dict["replacements"] = [TextNormalizationConfig.ReplacementObject.from_dict(replacement)
                                                  for replacement in self._config_dict["text_replacements"]]

    @property
    def replacements(self):
        return self.get_property("replacements")

    class ReplacementObject:
        def __init__(self, pattern, replace_text):
            self.pattern = pattern
            self.replace_text = replace_text

        @staticmethod
        def from_dict(replacement_dict):
            return TextNormalizationConfig.ReplacementObject(replacement_dict['pattern'], replacement_dict['replace'])


class VocabularyConfig(JsonConfig):
    CONFIG_FILENAME = 'vocabulary.json'

    def __init__(self):
        super(VocabularyConfig, self).__init__(VocabularyConfig.CONFIG_FILENAME)

    @property
    def inclusion_threshold(self):
        return self.get_property("inclusion_threshold")

    @property
    def trim_vocabulary(self):
        return self.get_property("trim_vocabulary")

    @property
    def start_of_sequence_token(self):
        return self.get_property("start_of_sequence_token")

    @property
    def end_of_sequence_token(self):
        return self.get_property("end_of_sequence_token")

    @property
    def unknown_word_token(self):
        return self.get_property("unknown_word_token")

    @property
    def pad_token(self):
        return self.get_property("pad_token")


class CorpusConfig(JsonConfig):
    CONFIG_FILENAME = 'corpus.json'

    def __init__(self):
        super(CorpusConfig, self).__init__(CorpusConfig.CONFIG_FILENAME)

    @property
    def corpus_folder(self):
        return self.get_property("corpus_folder")