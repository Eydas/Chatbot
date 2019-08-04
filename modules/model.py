import torch
from torch import nn
from modules.embedding import Embedding
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.loss import masked_nll_loss
from utils.teacher_forcing import TeacherForcing
from random import random


class Model(nn.Module):
    def __init__(self, vocabulary, training):
        super(Model, self).__init__()

        self._training = training
        self._embedding = Embedding(vocabulary = vocabulary)
        self._encoder = Encoder(embedding = self._embedding, training = training)
        self._decoder = Decoder(embedding = self._embedding, training = training)

        if training:
            self._teacher_forcing = TeacherForcing()
            # TODO: Look at other possible loss functions
            self._loss = masked_nll_loss


    def forward(self, input_seqs, input_lengths, target_seqs = None, masks = None, global_step = -1):
        if self._training:
            assert target_seqs is not None and masks is not None and global_step >= 0
            decoding_steps = target_seqs.shape[0]
        else:
            # TODO: Create model config for this and other settings
            decoding_steps = self._decoder._config.max_decoding_steps

        encoder_outputs, encoder_final_hidden_state = self._encoder(input_seqs, input_lengths)

        # Set decoder initial input to Start of Sequence token
        last_decoder_output = torch.tensor([self._embedding.vocabulary.start_of_sequence_token_index
                                             for _ in range(target_seqs.shape[1])])
        # TODO: Next line won't work if Encoder and Decoder RNN types don't match
        last_decoder_hidden_state = encoder_final_hidden_state

        if self._training:
            decoding_loss = 0.0
            print_loss = 0.0
            n_total = 0
        else:
            decoder_outputs = []

        for i in range(decoding_steps):
            last_decoder_output, last_decoder_hidden_state = self._decoder(last_decoder_output,
                                                                           last_decoder_hidden_state,
                                                                           encoder_outputs)

            if self._training:
                # Loss Computation
                # This calculates the mean loss over batches, but not over timesteps. Is this right?
                timestep_loss, timestep_n_total = self._loss(softmax_probs = last_decoder_output, targets = target_seqs[i], mask = masks[i])
                decoding_loss += timestep_loss
                print_loss = print_loss * n_total + timestep_loss * timestep_n_total
                n_total += timestep_n_total
                print_loss /= n_total

                if self._teacher_forcing.enabled and random() < self._teacher_forcing.get_current_ratio(global_step):
                    last_decoder_output = target_seqs[i]
                else:
                    # Choose last decoder output
                    last_decoder_output = torch.argmax(last_decoder_output, dim=1)

            else:
                # Greedy decoding. Are there other ways?
                last_decoder_output = torch.argmax(last_decoder_output, dim=1)
                decoder_outputs.append(last_decoder_output)

        if self._training:
            return decoding_loss, print_loss, None
        else:
            return None, None, decoder_outputs

'''
if __name__ == "__main__":
    from data_loading.corpus_loader import CORPUS_FILE, Corpus
    from data_loading.data_tensor_builder import BatchTensorBuilder
    from modules.loss import masked_nll_loss
    corpus = Corpus(CORPUS_FILE)
    tensor_builder = BatchTensorBuilder(corpus.seqs_data[:5], corpus.vocabulary)
    input_seqs = tensor_builder.input_seqs_tensor
    input_lengths = tensor_builder.input_lengths
    target_seqs = tensor_builder.target_seqs_tensor
    model = Model(corpus.vocabulary, training=True)
    decoder_outputs = model(input_seqs, input_lengths, target_seqs, 0)
    step_loss = masked_nll_loss(decoder_outputs, target_seqs, tensor_builder.masks)
    print(step_loss)
'''