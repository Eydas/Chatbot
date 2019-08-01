import torch
from torch import nn
from torch.nn import functional
from math import sqrt
from config_loading import AttentionConfig

class Attention(nn.Module):
    def __init__(self, query_size, key_size, value_size):
        super(Attention, self).__init__()

        self._config = AttentionConfig()

        self._score_func = {
            'dot': self._dot_score,
            'additive': self._additive_score,
            'general': self._general_score,
            'scaled_dot': self._scaled_dot_score
        }[self._config.score]

        self._init_params_func = {
            'dot': self._init_dot_params,
            'additive': self._init_additive_params,
            'general': self._init_general_params,
            'scaled_dot': self._init_scaled_dot_params
        }[self._config.score]

        self._query_size = query_size
        self._key_size = key_size
        self._value_size = value_size

        self._init_params()


    def forward(self, query, keys, values):
        # transpose to batch major
        query = torch.transpose(query, 0, 1)
        keys = torch.transpose(keys, 0, 1)
        values = torch.transpose(values, 0, 1)

        if self._config.project_query:
            query_projected = self._query_projection(query)
        else:
            query_projected = query

        if self._config.project_keys:
            keys_projected = self._key_projection(keys)
        else:
            keys_projected = keys

        if self._config.project_values:
            value_projected = self._value_projection(values)
        else:
            value_projected = values

        return self._score_func(query_projected, keys_projected, value_projected)


    def _init_params(self):
        if self._config.project_query:
            self._query_projection = nn.Linear(self._query_size, self._config.context_size)

        if self._config.project_keys:
            self._key_projection = nn.Linear(self._key_size, self._config.context_size)

        if self._config.project_values:
            self._value_projection = nn.Linear(self._value_size, self._config.context_size)

        self._init_params_func()


    def _init_dot_params(self):
        pass


    def _init_additive_params(self):
        self._additive_projection_matrix = nn.Linear(2 * self._config.context_size, self._config.context_size)
        # TODO: ensure tensor is saved on GPU when running with CUDA
        self._additive_projection_vector = nn.Parameter(torch.FloatTensor(self._config.context_size))


    def _init_general_params(self):
        self._general_projection = nn.Linear(self._config.context_size, self._config.context_size, bias=None)


    def _init_scaled_dot_params(self):
        pass


    def _apply_alignments(self, alignments, values):
        result = functional.softmax(alignments, dim=1)

        # Compute context vector
        result = result.unsqueeze(dim=1)
        result = result.bmm(values)

        return result


    def _dot_score(self, query, keys, values):
        alignments = torch.sum(query * keys, dim=2)
        return self._apply_alignments(alignments, values)


    def _additive_score(self, query, keys, values):
        alignments = torch.cat([query.expand(-1, keys.shape[1], -1), keys], dim=2)
        alignments = self._additive_projection_matrix(alignments).tanh()
        alignments = torch.sum(self._additive_projection_vector * alignments, dim=2)
        return self._apply_alignments(alignments, values)


    def _general_score(self, query, keys, values):
        alignments = torch.sum(query * self._general_projection(keys), dim=2)
        return self._apply_alignments(alignments, values)


    def _scaled_dot_score(self, query, keys, values):
        result = self._dot_score(query, keys, values)
        return result / sqrt(self._key_size)


if __name__ == "__main__":
    '''
    query = torch.tensor([[[2, 3]]]).type(torch.float32)
    keys = torch.tensor([[[4, 2]]]).type(torch.float32)
    values = torch.tensor([[[4, 2]]]).type(torch.float32)
    '''

    '''
    query = torch.tensor([[[2, 3]]]).type(torch.float32)
    keys = torch.tensor([[[4, 2]], [[4, 3]]]).type(torch.float32)
    values = torch.tensor([[[4, 2]], [[4, 3]]]).type(torch.float32)
    '''

    query = torch.tensor([[[2, 3], [4, 5]]]).type(torch.float32)
    keys = torch.tensor([[[4, 2], [1, 1]], [[4, 3], [2, 2]]]).type(torch.float32)
    values = torch.tensor([[[4, 2], [1, 1]], [[4, 3], [2, 2]]]).type(torch.float32)


    attn = Attention(2, 2, 2)
    weights = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]).type(torch.float32)
    bias = torch.tensor([-12, -11]).type(torch.float32)
    v = torch.tensor([1, 1]).type(torch.float32)
    attn._additive_projection_matrix.weight = nn.Parameter(weights)
    attn._additive_projection_matrix.bias = nn.Parameter(bias)
    attn._additive_projection_vector = nn.Parameter(v)
    res = attn(query, keys, values)
    print(res.shape)
    print(res)

