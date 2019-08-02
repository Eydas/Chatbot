from torch import nn
from config_loading import EmbeddingConfig

class Embedding(nn.Module):
    def __init__(self, vocabulary):
        super(Embedding, self).__init__()

        self._config = EmbeddingConfig()
        self.vocabulary = vocabulary
        self._embedding = nn.Embedding(num_embeddings = self.vocabulary.size,
                                       embedding_dim = self._config.embedding_size)

    def forward(self, token_index):
        # TODO: Add dropout?
        return self._embedding(token_index)

    @property
    def embedding_size(self):
        return self._config.embedding_size
