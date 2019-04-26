import torch
from corpus_loader import Corpus, CORPUS_FILE

class Embedder:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def embed_word(self, word):
        if word not in self.vocabulary.words:
            return torch.tensor([self.vocabulary.unknown_token_index])
        else:
            return torch.tensor([self.vocabulary.word_to_index(word)])

    def embed_sentence(self, sentence):
        words = sentence.strip('\n').strip().split(' ')
        return torch.cat([self.embed_word(word) for word in words] + [torch.tensor([self.vocabulary.end_of_sequence_token_index])])

    def embed_sequence_pairs(self, sequence_pairs):
        pass

if __name__ == "__main__":
    corpus = Corpus(CORPUS_FILE)
    data = corpus.seqs_data
    embedder = Embedder(corpus.vocabulary)
    print(data[0][0])
    print(embedder.embed_sentence(data[0][0]))
