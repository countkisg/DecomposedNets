from __future__ import print_function
from __future__ import absolute_import
import errno
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def mnist_hist(result, title):
    num_bins = 50

    fig = plt.figure(title)
    # the histogram of the data
    plt.hist(result, num_bins, normed=1, facecolor='green', alpha=0.5)
    plt.xlabel('Class')
    plt.ylabel('Counter')
    plt.title(title)
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    fig.show()
