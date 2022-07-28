# import os
# import pickle

# actors= os.listdir('data')

# #putting the path of each image in a list called filenames
# filenames=[]
# for actor in actors:
#     for files in os.listdir(os.path.join('data',actor)):
#         filenames.append(os.path.join('data',actor,files))

# pickle.dump(filenames,open('filenames.pkl','wb'))

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm 

filenames=pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def feature_extractor(path,model):
    img=image.load_img(path,target_size=(224,224))
    img_array = image.img_to_array(img)#The img_to_array() function adds channels
    expanded_img=np.expand_dims(img_array,axis=0)#is used to add the number of images: x.shape = (1, 224, 224, 3):
    preprocessed_img = preprocess_input(expanded_img)
    #preprocess_input subtracts the mean RGB channels of the imagenet dataset. This is because the model you are using has been trained on a different dataset

    result=model.predict(preprocessed_img).flatten()
    return result

features=[]
for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))


