from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm
import pickle

filenames=pickle.load(open('filenames.pkl','rb'))
model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def feature_extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expended_img=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expended_img)

    result=model.predict(preprocessed_img).flatten()
    return result

features=[]

#tqdm for faster iteration

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embeddings.pkl','wb'))
