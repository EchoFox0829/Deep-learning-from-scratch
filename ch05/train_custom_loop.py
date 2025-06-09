# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import time
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# Hyperparameter settings
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5  # Time size for Truncated BPTT
lr = 0.1
max_epoch = 100

# Load training data (reduce dataset size)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # Input
ts = corpus[1:]  # Output (teacher labels)
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

# Variables used in training
max_iters = data_size // (batch_size * time_size)
iters = 0
epoch = 0
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# Generate model
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# Calculate starting position for each sample in mini-batch
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iters in range(max_iters):
        # Get mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx + 1) % data_size]
            time_idx += 1

        # Calculate gradient and update parameters
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # Evaluate perplexity for each epoch
    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# Plot graph
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()