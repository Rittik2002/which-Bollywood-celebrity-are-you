import os
import pickle
celebs=os.listdir('celeb')

filenames=[]

for celeb in celebs:
    for file in os.listdir(os.path.join('celeb',celeb)):
        filenames.append(os.path.join('celeb',celeb,file))

pickle.dump(filenames,open('filenames.pkl','wb'))
# type: ignore