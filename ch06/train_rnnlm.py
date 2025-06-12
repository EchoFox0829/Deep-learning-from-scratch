# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm


# Hyperparameter settings
batch_size = 20
wordvec_size = 100
hidden_size = 100  # Number of elements in RNN hidden state vector
time_size = 35  # Size to unroll RNN
lr = 20.0
max_epoch = 4
max_grad = 0.25

# Load training data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# Generate model
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# Apply gradient clipping and train
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad)
trainer.plot(ylim=(0, 500))

# Evaluate on test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# Save parameters
model.save_params()