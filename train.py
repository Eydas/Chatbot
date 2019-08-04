import torch
from torch import nn, optim
from config_loading import TrainConfig
from modules.model import Model
from modules.loss import masked_nll_loss
from os import path, makedirs
from data_loading.corpus import CorpusDataset
import logging

class Train:
    def __init__(self, model_name, corpus_dataset):
        self._config = TrainConfig()
        self._model_name = model_name
        self._data_loader = corpus_dataset.get_data_loader(self._config.batch_size)
        self._vocabulary = corpus_dataset.vocabulary
        self._model = Model(vocabulary = corpus_dataset.vocabulary, training = True)
        self._loss = masked_nll_loss
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._config.learning_rate)
        self._global_step = -1

        self._train_logger = logging.getLogger('Train')
        logging.basicConfig(level=logging.INFO)


    def train_step(self, input_seqs, input_lengths, target_seqs, masks):
        self._optimizer.zero_grad()
        decoder_outputs = self._model(input_seqs, input_lengths, target_seqs, self._global_step)
        step_loss = self._loss(decoder_outputs, target_seqs, masks)

        self._train_logger.info('Step {}:  Training loss: {}'.format(self._global_step, step_loss.item()))

        step_loss.backward()

        # TODO: Gradient clipping HERE
        if self._config.use_gradient_clipping:
            _ = nn.utils.clip_grad_norm_(self._model.parameters(), self._config.gradient_clipping_value)

        self._optimizer.step()


    def train(self, num_steps, save_num_steps, save_folder = './data/models/train_dev'):

        if self._global_step < 0:
            self._global_step = 0
        elif self._global_step >= num_steps:
            logging.info('Global step past number of steps requested. No training needed. Global Step = {}. '
                         'Num training steps = {}'.format(self._global_step, num_steps))
            return

        stop_training = False

        while not stop_training:
            for input_seqs, input_lengths, target_seqs, masks in self._data_loader:
                self.train_step(input_seqs, input_lengths, target_seqs, masks)
                self._global_step += 1

                if self._global_step % save_num_steps == 0:
                    self.save_checkpoint(save_folder)
                    just_saved = True
                else:
                    just_saved = False

                if self._global_step >= num_steps:
                    stop_training = True
                    logging.info('Finished training at step {}'.format(self._global_step))
                    if not just_saved:
                        self.save_checkpoint(save_folder)
                    break


    def save_checkpoint(self, save_folder):
        makedirs(save_folder, exist_ok=True)
        save_path = path.join(save_folder, 'checkpoint-{}.tar'.format(self._global_step))
        logging.info('Saving checkpoint at step {}'.format(self._global_step))
        torch.save(
            {
                'name': self._model_name,
                'global_step': self._global_step,
                'model': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'vocabulary': self._vocabulary.__dict__,
            }, save_path)
        logging.info('Checkpoint saved at {}'.format(save_path))

    @staticmethod
    def load_from_checkpoint(checkpoint_path, corpus_dataset):
        checkpoint = torch.load(checkpoint_path)
        train_obj = Train(checkpoint['name'], corpus_dataset)
        train_obj._vocabulary.__dict__ = checkpoint['vocabulary']
        train_obj._global_step = checkpoint['global_step']
        train_obj._model.load_state_dict(checkpoint['model'])
        train_obj._train_logger.info('Restored from checkpoint {}'.format(checkpoint_path))
        return train_obj





if __name__ == "__main__":
    from data_loading.corpus_loader import CORPUS_FILE
    corpus_filepath = path.join('./data/corpus', CORPUS_FILE)
    save_folder = '/media/Work/data/Chatbot/models/train_dev'
    train_obj = Train("train_dev", CorpusDataset(corpus_filepath))
    train_obj.train(num_steps = 4000, save_num_steps = 500, save_folder = save_folder)
    #train_obj = Train.load_from_checkpoint(path.join(save_folder, 'checkpoint-20.tar'), CorpusDataset(corpus_filepath))
    #train_obj.train(num_steps= 20, save_num_steps= 10, save_folder = save_folder)