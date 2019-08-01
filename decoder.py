from config_loading import EncoderConfig, DecoderConfig, AttentionConfig
from attention import Attention
import torch
from torch import nn
from torch.nn import functional
import numpy as np

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


    def forward(self, encoder_outputs, encoder_final_state, targets = None):
        if self._training:
            assert targets is not None
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
            # TODO: Teacher forcing here
            last_decoder_output = torch.tensor(torch.argmax(last_decoder_output, dim=1))

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs


    def _bahdanau_step(self, last_decoder_output, last_decoder_hidden_state, encoder_outputs):
        last_output_embedded = self._embedding(last_decoder_output).unsqueeze(0)
        query = last_decoder_hidden_state[-1].unsqueeze(0)
        context = self._attention(query, encoder_outputs, encoder_outputs)

        # remove time dim from rnn_output and context vector
        last_output_embedded = last_output_embedded.squeeze(0)
        context = context.squeeze(1)

        rnn_input = torch.cat((last_output_embedded, context), -1)
        rnn_output, rnn_hidden_state = self._rnn(rnn_input, last_decoder_hidden_state)

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


if __name__ == "__main__":
    from corpus_loader import CORPUS_FILE, Corpus
    from data_tensor_builder import BatchTensorBuilder
    from embedding import Embedding
    from encoder import Encoder
    corpus = Corpus(CORPUS_FILE)
    tensor_builder = BatchTensorBuilder(corpus.seqs_data[:5], corpus.vocabulary)
    input_seqs = tensor_builder.input_seqs_tensor
    input_lengths = tensor_builder.input_lengths
    target_seqs = tensor_builder.target_seqs_tensor
    embedding = Embedding(corpus.vocabulary)
    encoder = Encoder(embedding, training=True)
    encoder_outputs, encoder_final_hidden_state = encoder(input_seqs, input_lengths)
    decoder = Decoder(embedding, training=True)
    decoder_outputs = decoder(encoder_outputs, encoder_final_hidden_state, target_seqs)
    print(decoder_outputs.shape)
    print(target_seqs.shape)
    print(decoder_outputs)

