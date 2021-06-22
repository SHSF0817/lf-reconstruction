import os
import random

root = '/home/B/hx_data/DLFD_train/'
test_num = 9
dirs = os.listdir(root)
random.shuffle(dirs)

for i in range(test_num):
    dir_path = root+dirs[i]
    os.system('mv '+ dir_path + ' /home/B/hx_data/DLFD_test/')
    print(dir_path)