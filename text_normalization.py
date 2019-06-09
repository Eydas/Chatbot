import re
from config_loading import TextNormalizationConfig

class TextNormalizer:
    def __init__(self):
        self._replacements = TextNormalizationConfig().replacements

    def normalize_text(self, text):
        text = text.lower()

        for replacement in self._replacements:
            text = re.sub(replacement.pattern, replacement.replace_text, text)

        return text