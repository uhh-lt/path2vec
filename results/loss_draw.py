#!/usr/bin/python3
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt


values = {}

for line in sys.stdin:
    if line.strip().startswith('#'):
        continue
    res = line.strip().split(',')
    (model, epoch, loss) = res
    if model.strip() not in values:
        values[model] = {'epochs': [], 'losses': []}
    model = model.strip()
    epoch = int(epoch.strip())
    loss = float(loss.strip())
    values[model]['epochs'].append(epoch)
    values[model]['losses'].append(loss)

plt.figure()

for model in values:
    epochs = np.array(values[model]['epochs'])
    losses = np.array(values[model]['losses'])
    plt.plot(epochs, losses, linestyle='dashed', marker='o', label=model)

plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.legend(loc='best')
plt.grid(True)
plt.title('Models learning curves across epochs')
plt.show()
# plt.savefig(corpus + '_static_synsets.png', dpi=300)
plt.close()

