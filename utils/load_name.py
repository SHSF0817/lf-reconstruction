import os
import pickle

filenames = os.listdir('/home/B/huangxin_data/full_data/additional/cotton')
filenames = list(filter(lambda filename: os.path.splitext(filename)[0][0:5] == 'input', filenames))
filenames.sort()
with open('filenames.dat', 'wb') as file:
    pickle.dump(filenames,file)

filenames = pickle.load(open('filenames.dat','rb'))

print([filenames]*3)