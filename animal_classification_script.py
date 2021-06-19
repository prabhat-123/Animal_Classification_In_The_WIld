import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tqdm import tqdm
import shutil
from animal_classification_report_generator import Animal_Classification_Report

# root_dir = os.getcwd()

tf.keras.backend.clear_session()

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# load the model from json file
json_file = open('animal_classifier_0.4dropoutxceptionnet.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

# Load the weights into a model

model.load_weights('animal_classifier_xceptionnet_05-0.925932.h5')
print("Model Loaded Successfully")


def model_predict(image_path,model):
    img = image.load_img(image_path,target_size=(300,300))
    # print(img)
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    # Rescaling the image
    x = x/255.0
    preds = model.predict(x)
    return preds



folder_path = input("Please provide the path where your folder containing images are present")
folder_name = folder_path.split('/')[-1]
if not os.path.exists(os.path.join(folder_path,folder_name + '_' + 'Animals')):
    os.mkdir(os.path.join(folder_path,folder_name+'_'+'Animals'))
else: 
    pass
if not os.path.exists(os.path.join(folder_path, folder_name + '_' + 'No_Animals')):
    os.mkdir(os.path.join(folder_path,folder_name+'_'+'No_Animals'))
else:
    pass

animal_folder_path = os.path.join(folder_path,folder_name+'_'+'Animals')
no_animal_folder_path = os.path.join(folder_path,folder_name+'_'+'No_Animals')
csv_name = os.path.join(folder_path,folder_name + '_' + 'report' + '.csv')
animal_classification_report_obj = Animal_Classification_Report(csv_filename=csv_name)
animal_classification_report_obj.write_csv_header(image_name="Image_Name",original_image_path="Original_Image_Path",generated_image_path="Generated_Image_Path",prediction="Prediction",confidence="Confidience")

for img in tqdm(os.listdir(folder_path)):
    if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.JPG'):
        img_path = os.path.join(folder_path,img)
        try:
            preds = model_predict(img_path,model)
            if preds > 0.5:
                pos_conf = preds * 100
                neg_conf = 100 - pos_conf
                pred_class = "No_Animal"
                non_pred_class = "Animal"
                shutil.copy(img_path,os.path.join(no_animal_folder_path,img))
                animal_classification_record = [img,img_path,os.path.join(no_animal_folder_path,img),pred_class,pos_conf[0][0]]
                animal_classification_report_obj.append_csv_rows(records=animal_classification_record)
            else:
                pos_conf = 50 + (0.5-preds) * 100
                neg_conf = 100 - pos_conf
                pred_class = "Animal"
                non_pred_class = "No_Animal"
                shutil.copy(img_path,os.path.join(animal_folder_path,img))
                animal_classification_record = [img,img_path,os.path.join(animal_folder_path,img),pred_class,pos_conf[0][0]]
                animal_classification_report_obj.append_csv_rows(records=animal_classification_record)
        except Exception as e:
            pass
        
