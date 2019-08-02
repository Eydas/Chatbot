from torch import nn, optim
from config_loading import TrainConfig
from modules.model import Model
from modules.loss import masked_nll_loss
from data_loading.data_tensor_builder import BatchTensorBuilder
import logging

class Train:
    def __init__(self, corpus):
        self._config = TrainConfig()
        self._feeder = BatchTensorBuilder(corpus.seqs_data, corpus.vocabulary)
        self._model = Model(vocabulary = corpus.vocabulary, training = True)
        self._loss = masked_nll_loss
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._config.learning_rate)
        self._global_step = 0

        self._train_logger = logging.getLogger('Train')
        self._train_logger.setLevel(logging.INFO)

        #logging.getLogger().setLevel(logging.INFO)

    def train_step(self, input_seqs, input_lengths, target_seqs, masks):
        self._optimizer.zero_grad()
        decoder_outputs = self._model(input_seqs, input_lengths, target_seqs, self._global_step)
        step_loss = self._loss(decoder_outputs, target_seqs, masks)

        # TODO: Print Loss HERE
        self._train_logger.info('Step {}:  Training loss: {}'.format(self._global_step, step_loss.item()))

        step_loss.backward()

        # TODO: Gradient clipping HERE
        if self._config.use_gradient_clipping:
            _ = nn.utils.clip_grad_norm_(self._model.parameters(), self._config.gradient_clipping_value)

        self._optimizer.step()

    def train(self, num_steps):
        for i in range(num_steps):
            self._global_step = i
            self.train_step(self._feeder.input_seqs_tensor, self._feeder.input_lengths, self._feeder.target_seqs_tensor, self._feeder.masks)


if __name__ == "__main__":
    from data_loading.corpus_loader import CORPUS_FILE, Corpus
    train_obj = Train(Corpus(CORPUS_FILE))
    train_obj.train(1000)