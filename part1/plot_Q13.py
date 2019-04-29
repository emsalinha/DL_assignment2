import csv
import matplotlib.pyplot as plt
import numpy as np
from part1.open_results import open_results
save = True
model = 'RNN'

results, T, accuracies, n_epochs = open_results(model)


plt.gca().set_color_cycle(['green'])

plt.plot(T, accuracies)

plt.legend(['accuracy'], loc='lower left')
plt.ylabel('accuracy')
plt.xlabel('palindrome length')
plt.ylim(0,1.2)

if save:
    plt.savefig('accuracy_{}'.format(model))
    plt.close()
else:
    plt.show()


plt.gca().set_color_cycle(['blue'])
plt.plot(T, n_epochs)
plt.legend(['epochs'], loc='upper left')
plt.ylabel('epochs for convergence')
plt.xlabel('palindrome length')

if save:
    plt.savefig('epochs_{}'.format(model))
else:
    plt.show()

