from config_loading import EncoderConfig, DecoderConfig
from attention import Attention
import torch
from torch import nn
from torch.nn import functional

class Decoder(nn.Module):

    # TODO: Unify Encoder and Decoder RNN Factory
    RNN_FACTORY = {
        'GRU': nn.GRU,
        'LSTM': nn.LSTM
    }

    def __init__(self, embedding, training):
        super(Decoder, self).__init__()
        self._config = DecoderConfig()
        self._training = training

        self._key_size = EncoderConfig().hidden_size
        self._value_size = self._key_size

        self._embedding = embedding

        self._forward_step = {
            'bahdanau' : self._bahdanau_step,
            'luong': self._luong_step
        }[self._config.attention_mechanism]

        self._attention = Attention(score_type = self._config.attention_score,
                                    project = self._config.attention_projection,
                                    context_size = self._config.context_size,
                                    query_size = self._config.hidden_size,
                                    key_size = self._key_size,
                                    value_size = self._value_size)

        self._rnn = Decoder.RNN_FACTORY[self._config.rnn_type](
            input_size = self._embedding.embedding_size,
            hidden_size = self._config.hidden_size,
            num_layers = self._config.num_layers,
            bidirectional = False,
            bias = self._config.rnn_bias,
            dropout = (self._config.dropout_probability if self._training and self._config.dropout_enabled else 0)
        )

        self._mlp = nn.Linear(self._config.hidden_size, self._embedding.vocabulary.size)

    def forward(self, encoder_outputs, encoder_final_state, targets = None):
        if self._training:
            assert targets is not None
            decoding_steps = targets.shape[0] - 1
        else:
            decoding_steps = self._config.max_decoding_steps

        last_decoder_output = torch.tensor([self._embedding.vocabulary.start_of_sequence_index])
        # TODO: Next line won't work if Encoder and Decoder RNN types don't match
        last_decoder_hidden_state = encoder_final_state
        decoder_outputs = []

        for i in range(decoding_steps):
            last_decoder_output, last_decoder_hidden_state = self._forward_step(last_decoder_output,
                                                                                last_decoder_hidden_state,
                                                                                encoder_outputs)
            decoder_outputs.append(last_decoder_output)
            # TODO: Teacher forcing here

        return decoder_outputs


    def _bahdanau_step(self, last_decoder_output, last_decoder_hidden_state, encoder_outputs):
        return None, None


    def _luong_step(self, last_decoder_output, last_decoder_hidden_state, encoder_outputs):
        last_output_embedded = self._embedding(last_decoder_output).unsqueeze(0)
        rnn_output, rnn_hidden_state = self._rnn(last_output_embedded, last_decoder_hidden_state)
        context = self._attention(rnn_output, encoder_outputs, encoder_outputs)

        # remove time dim from rnn_output and context vector
        rnn_output = rnn_output.squeeze(0)
        context_vector = context.squeeze(1)

        # MLP on concatenated rnn output and context vector to produce softmax probabilities over tokens
        concat_input = torch.cat((rnn_output, context), 1)
        concat_activated = torch.tanh(self.concat(concat_input))
        output = self._mlp(concat_activated)
        output = functional.softmax(output, dim=1)

        return output, rnn_hidden_state
