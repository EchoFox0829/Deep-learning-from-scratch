# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# Hyperparameter settings
batch_size = 10
wordvec_size = 100
hidden_size = 100  # Number of elements in RNN hidden state vector
time_size = 5  # Size to unroll RNN
lr = 0.1
max_epoch = 100

# Load training data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000  # Reduce dataset size for testing
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]  # Input
ts = corpus[1:]  # Output (teacher labels)

# Generate model
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()