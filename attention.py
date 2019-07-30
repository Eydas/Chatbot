import torch
from torch import nn
from torch.nn import functional
from math import sqrt

class Attention(nn.Module):
    def __init__(self, score_type, project, context_size, query_size, key_size, value_size):
        super(Attention, self).__init__()

        self._score_func = {
            'dot': self._dot_score,
            'additive': self._additive_score,
            'general': self._general_score,
            'scaled_dot': self._scaled_dot_score
        }[score_type]

        self._init_params_func = {
            'dot': self._init_dot_params,
            'additive': self._init_additive_params,
            'general': self._init_general_params,
            'scaled_dot': self._init_scaled_dot_params
        }[score_type]

        self._project = project
        self._context_size = context_size
        self._query_size = query_size
        self._key_size = key_size
        self._value_size = value_size

        self._init_params()


    def forward(self, query, keys, values):
        # transpose to batch major
        query = torch.transpose(query, 0, 1)
        keys = torch.transpose(keys, 0, 1)
        values = torch.transpose(values, 0, 1)

        if self._project:
            query_projected = self._query_projection(query)
            keys_projected = self._key_projection(keys)
            value_projected = self._value_projection(values)
            return self._score_func(query_projected, keys_projected, value_projected)
        else:
            return self._score_func(query, keys, values)


    def _init_params(self):
        if self._project:
            self._query_projection = nn.Linear(self._query_size, self._context_size)
            self._key_projection = nn.Linear(self._key_size, self._context_size)
            self._value_projection = nn.Linear(self._value_size, self._context_size)

        self._init_params_func()


    def _init_dot_params(self):
        pass


    def _init_additive_params(self):
        self._additive_projection_matrix = nn.Linear(2 * self._context_size, self._context_size)
        # TODO: ensure tensor is saved on GPU when running with CUDA
        self._additive_projection_vector = nn.Parameter(torch.FloatTensor(self._context_size))


    def _init_general_params(self):
        self._general_projection = nn.Linear(self._context_size, self._context_size, bias=None)


    def _init_scaled_dot_params(self):
        pass


    def _apply_alignments(self, alignments, values):
        #result = functional.softmax(alignments, dim=1)
        result = alignments

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


    attn = Attention('additive', False, 2, 2, 2, 2)
    weights = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]).type(torch.float32)
    bias = torch.tensor([-12, -11]).type(torch.float32)
    v = torch.tensor([1, 1]).type(torch.float32)
    attn._additive_projection_matrix.weight = nn.Parameter(weights)
    attn._additive_projection_matrix.bias = nn.Parameter(bias)
    attn._additive_projection_vector = nn.Parameter(v)
    res = attn(query, keys, values)
    print(res.shape)
    print(res)

