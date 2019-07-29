from torch import nn
from config_loading import EncoderConfig
from corpus_loader import Corpus, CORPUS_FILE
from data_tensor_builder import BatchTensorBuilder


class Encoder(nn.Module):

    RNN_FACTORY = {
        'GRU': nn.GRU,
        'LSTM': nn.LSTM
    }

    def __init__(self):
        super(Encoder, self).__init__()
        self._config = EncoderConfig()

        self._rnn = Encoder.RNN_FACTORY[self._config.rnn_type](
            input_size = self._config.hidden_size,#TODO: This should be embedding_size
            hidden_size = self._config.hidden_size,
            num_layers = self._config.num_layers,
            bidirectional = self._config.bidirectional,
            bias = self._config.bias,
            dropout = (self._config.dropout_probability if self._config.dropout_enabled else 0)
        )


    def forward(self, input_seqs, input_lengths, initial_hidden_state = None):
        # Why shouldn't the batch be first? Also, enforce_sorted may be more efficient
        packed_input_seqs = nn.utils.rnn.pack_padded_sequence(input_seqs,
                                                              input_lengths,
                                                              batch_first=False,
                                                              enforce_sorted=False)

        packed_rnn_outputs, rnn_final_state = self._rnn(packed_input_seqs, initial_hidden_state)

        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_rnn_outputs, batch_first=False)

        rnn_outputs = self._finalize_rnn_outputs(rnn_outputs)
        rnn_final_state = self._finalize_rnn_final_state(rnn_final_state)

        return rnn_outputs, rnn_final_state


    def _finalize_rnn_outputs(self, rnn_outputs):
        if self._config.bidirectional:
            # sum forwards and backwards hidden states. TODO: Add option to leave concatenated
            return rnn_outputs[:, :, :self._config.hidden_size] + rnn_outputs[:, :, self._config.hidden_size:]
        else:
            return rnn_outputs


    def _finalize_rnn_final_state(self, rnn_final_state):
        # Remove cell state to keep consistent with GRU. Is there a better way?
        if self._config.rnn_type == 'LSTM':
            rnn_final_state, _ = rnn_final_state

        #TODO: Deal with bidirectional and number of layers? Or keep as is?
        if self._config.bidirectional:
            rnn_final_state = rnn_final_state.view(self._config.num_layers, 2, rnn_final_state.shape[1], rnn_final_state.shape[2])
            # sum forwards and backwards final states. TODO: Add option to leave concatenated
            rnn_final_state = rnn_final_state[:,0,:,:] + rnn_final_state[:,1,:,:]

        return rnn_final_state

if __name__ == '__main__':
    corpus = Corpus(CORPUS_FILE)
    tensor_builder = BatchTensorBuilder(corpus.seqs_data[:5], corpus.vocabulary)
    input_seqs = tensor_builder.input_seqs_tensor
    input_lengths = tensor_builder.input_lengths
    encoder = Encoder()
    input_seqs = nn.Embedding(num_embeddings=corpus.vocabulary.size,embedding_dim=encoder._config.hidden_size)(input_seqs)
    encoder_outputs, final_hidden_state = encoder(input_seqs, input_lengths)
    print(encoder_outputs.shape)
    print(final_hidden_state.shape)
    print(encoder_outputs)
    print(final_hidden_state)