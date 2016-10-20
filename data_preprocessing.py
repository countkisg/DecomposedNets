import scipy.misc
import numpy as np
from os import listdir
from os.path import isfile, join
data_directory = '__img_align_celeba/'
onlyfiles = [data_directory + f for f in listdir(data_directory) if
                   isfile(join(data_directory, f))]
