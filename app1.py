# -*- coding: utf-8 -*-
"""
Created on Mon May 10 00:55:20 2021

@author: Rutik Desai
"""

from __future__ import division, print_function
import streamlit as st
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
#from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
#app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='maleria_detection_vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)





def model_predict(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)

    preds=np.argmax(preds, axis=1)

    if preds==0:
        preds="The Person is Infected With Pneumonia"
    else:
        preds="The Person is not Infected With Pneumonia"
    
    
    return preds





def main():
    st.title("Malaria Detection")
    f=st.file_uploader("Upload Image",type=["png","jpg","jpeg"])
    if f is not None:
        with open(os.path.join("uploads",f.name),"wb") as n:
            n.write(f.getbuffer())
        st.success("File Saved")
    
 
    result=""
    if st.button("Predict"):
        
        preds = model_predict(os.path.join("uploads",f.name),model)
        result=preds
    st.success(f"{result}")
    return result
    


    

if __name__ == '__main__':
    main()
