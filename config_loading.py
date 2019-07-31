import json
from os import path

# TODO: Automatic validation of config values

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
        return self.get_property("start_of_sequence_token")["token"]

    @property
    def start_of_sequence_index(self):
        return self.get_property("start_of_sequence_token")["index"]

    @property
    def end_of_sequence_token(self):
        return self.get_property("end_of_sequence_token")["token"]

    @property
    def end_of_sequence_index(self):
        return self.get_property("end_of_sequence_token")["index"]

    @property
    def unknown_token(self):
        return self.get_property("unknown_token")["token"]

    @property
    def unknown_index(self):
        return self.get_property("unknown_token")["index"]

    @property
    def pad_token(self):
        return self.get_property("pad_token")["token"]

    @property
    def pad_index(self):
        return self.get_property("pad_token")["index"]


class CorpusConfig(JsonConfig):
    CONFIG_FILENAME = 'corpus.json'

    def __init__(self):
        super(CorpusConfig, self).__init__(CorpusConfig.CONFIG_FILENAME)

    @property
    def corpus_folder(self):
        return self.get_property("corpus_folder")


class EmbeddingConfig(JsonConfig):
    CONFIG_FILENAME = 'embedding.json'

    def __init__(self):
        super(EmbeddingConfig, self).__init__(EmbeddingConfig.CONFIG_FILENAME)

    @property
    def embedding_size(self):
        return self.get_property("embedding_size")


class EncoderConfig(JsonConfig):
    CONFIG_FILENAME = 'encoder.json'

    def __init__(self):
        super(EncoderConfig, self).__init__(EncoderConfig.CONFIG_FILENAME)

    @property
    def rnn_type(self):
        return self.get_property("rnn_type")

    @property
    def bidirectional(self):
        return self.get_property("bidirectional")

    @property
    def num_layers(self):
        return self.get_property("num_layers")

    @property
    def hidden_size(self):
        return self.get_property("hidden_size")

    @property
    def rnn_bias(self):
        return self.get_property("rnn_bias")

    @property
    def dropout_enabled(self):
        return self.get_property("dropout_enabled")

    @property
    def dropout_probability(self):
        return self.get_property("dropout_probability")


class DecoderConfig(JsonConfig):
    CONFIG_FILENAME = 'decoder.json'

    def __init__(self):
        super(DecoderConfig, self).__init__(DecoderConfig.CONFIG_FILENAME)

    @property
    def max_decoding_steps(self):
        return self.get_property("max_decoding_steps")

    @property
    def attention_mechanism(self):
        return self.get_property("attention_mechanism")

    @property
    def attention_score(self):
        return self.get_property("attention_score")

    @property
    def attention_projection(self):
        return self.get_property("attention_projection")

    @property
    def context_size(self):
        return self.get_property("context_size")

    @property
    def rnn_type(self):
        return self.get_property("rnn_type")

    @property
    def num_layers(self):
        return self.get_property("num_layers")

    @property
    def hidden_size(self):
        return self.get_property("hidden_size")

    @property
    def rnn_bias(self):
        return self.get_property("rnn_bias")

    @property
    def dropout_enabled(self):
        return self.get_property("dropout_enabled")

    @property
    def dropout_probability(self):
        return self.get_property("dropout_probability")