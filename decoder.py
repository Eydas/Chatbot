from config_loading import EncoderConfig, DecoderConfig, AttentionConfig
from attention import Attention
from teacher_forcing import TeacherForcing
import torch
from torch import nn
from torch.nn import functional
from random import random

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
        self._context_size = AttentionConfig().context_size

        self._embedding = embedding

        self._forward_step = {
            'bahdanau' : self._bahdanau_step,
            'luong': self._luong_step
        }[self._config.attention_mechanism]

        self._attention = Attention(query_size = self._config.hidden_size,
                                    key_size = self._key_size,
                                    value_size = self._value_size)

        rnn_input_size = {
            'bahdanau': self._embedding.embedding_size + self._context_size,
            'luong': self._embedding.embedding_size
        }[self._config.attention_mechanism]

        mlp_input_size = {
            'bahdanau': self._embedding.embedding_size + self._config.hidden_size + self._context_size,
            'luong': self._embedding.embedding_size + self._context_size
        }[self._config.attention_mechanism]

        self._rnn = Decoder.RNN_FACTORY[self._config.rnn_type](
            input_size = rnn_input_size,
            hidden_size = self._config.hidden_size,
            num_layers = self._config.num_layers,
            bidirectional = False,
            bias = self._config.rnn_bias,
            dropout = (self._config.rnn_dropout_probability
                       if self._training and self._config.rnn_dropout_enabled
                       else 0)
        )

        self._mlp = nn.Sequential(
            nn.Linear(mlp_input_size, self._config.hidden_size),
            nn.Tanh(),
            nn.Linear(self._config.hidden_size, self._embedding.vocabulary.size)
        )

        self._teacher_forcing = TeacherForcing()


    def forward(self, encoder_outputs, encoder_final_state, targets = None, global_step = -1):
        if self._training:
            assert targets is not None and global_step >= 0
            decoding_steps = targets.shape[0]
        else:
            decoding_steps = self._config.max_decoding_steps

        last_decoder_output = torch.tensor([self._embedding.vocabulary.start_of_sequence_token_index
                                             for _ in range(targets.shape[1])])
        # TODO: Next line won't work if Encoder and Decoder RNN types don't match
        last_decoder_hidden_state = encoder_final_state
        decoder_outputs = []

        for i in range(decoding_steps):
            last_decoder_output, last_decoder_hidden_state = self._forward_step(last_decoder_output,
                                                                                last_decoder_hidden_state,
                                                                                encoder_outputs)
            decoder_outputs.append(last_decoder_output)

            # Teacher forcing
            if self._training and self._teacher_forcing.enabled \
                    and random() < self._teacher_forcing.get_current_ratio(global_step):
                last_decoder_output = targets[i]
                print("Teacher forcing: {}".format(last_decoder_output))
            else:
                # Greedy decoding. Are there other ways?
                last_decoder_output = torch.argmax(last_decoder_output, dim=1)
                print("Not teacher forcing: {}".format(last_decoder_output))

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs


    def _bahdanau_step(self, last_decoder_output, last_decoder_hidden_state, encoder_outputs):
        last_output_embedded = self._embedding(last_decoder_output).unsqueeze(0)
        query = last_decoder_hidden_state[-1].unsqueeze(0)
        context = self._attention(query, encoder_outputs, encoder_outputs)

        # change context shape to time major for RNN
        context = torch.transpose(context, 0, 1)

        rnn_input = torch.cat((last_output_embedded, context), -1)
        rnn_output, rnn_hidden_state = self._rnn(rnn_input, last_decoder_hidden_state)

        # remove time dim from embedded rnn input, rnn_output and context vector
        last_output_embedded = last_output_embedded.squeeze(0)
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(0)

        mlp_input = torch.cat((last_output_embedded, rnn_output, context), -1)
        output = self._mlp(mlp_input)
        output = functional.softmax(output, dim=1)

        return output, rnn_hidden_state


    def _luong_step(self, last_decoder_output, last_decoder_hidden_state, encoder_outputs):
        last_output_embedded = self._embedding(last_decoder_output).unsqueeze(0)
        rnn_output, rnn_hidden_state = self._rnn(last_output_embedded, last_decoder_hidden_state)
        context = self._attention(rnn_output, encoder_outputs, encoder_outputs)

        # remove time dim from rnn_output and context vector
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        # MLP on concatenated rnn output and context vector to produce softmax probabilities over tokens
        concat_input = torch.cat((rnn_output, context), 1)
        output = self._mlp(concat_input)
        output = functional.softmax(output, dim=1)

        return output, rnn_hidden_state

