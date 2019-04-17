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


class RawDataProcessingConfig(JsonConfig):
    CONFIG_FILENAME = 'raw_data_processing.json'

    def __init__(self):
        super(RawDataProcessingConfig, self).__init__(RawDataProcessingConfig.CONFIG_FILENAME)

    @property
    def conversation_separator(self):
        return  self.get_property("conversation_separator")