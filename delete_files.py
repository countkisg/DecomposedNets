from os import listdir, remove
from os.path import isfile, join
import numpy as np
data_directory = 'img_align_celeba/'
onlyfiles = np.sort([data_directory + f for f in listdir(data_directory) if isfile(join(data_directory, f)) & (f>'202496.jpg')])
for i in range(len(onlyfiles)):
    remove(onlyfiles[i])
    print 'remove: %s \n' % onlyfiles[i]
