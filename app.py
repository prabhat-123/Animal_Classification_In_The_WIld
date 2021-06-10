import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from flask import Flask,url_for,request,render_template,redirect
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename


tf.keras.backend.clear_session()

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# Define a flask app
app = Flask(__name__,static_folder='./static')



# Type of Chest-xray cases that we are going to predict

# categories = ['Animals','No_Animals']

# load the model from json file
json_file = open('animal_classifier_0.4dropoutxceptionnet.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

# Load the weights into a model

model.load_weights('animal_classifier_xceptionnet_05-0.925932.h5')
print("Model Loaded Successfully")

#
model.summary()
# model._make_predict_function()

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

@app.route('/',methods=['GET'])
def index():
    return render_template('./index.html')


@app.route('/',methods=['POST','GET'])
def upload():
    if request.method=='POST':
        f = request.files['file']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path,'static','uploads',secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path,model)
        if preds > 0.5:
            pos_conf = preds * 100
            neg_conf = 100 - pos_conf
            pred_class = "No_Animal"
            non_pred_class = "Animal"
        else:
            pos_conf = 50 + (0.5-preds) * 100
            neg_conf = 100 - pos_conf
            pred_class = "Animal"
            non_pred_class = "No_Animal"
        return render_template('./predict.html',result=pred_class,support=non_pred_class,image_url = f.filename,positive_confidence = pos_conf,negative_confidence = neg_conf)
    else:
        return render_template('./index.html')

if __name__ == '__main__':
    app.run(debug=True)

