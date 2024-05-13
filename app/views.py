import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as matimg
import sqlite3
import pandas as pd


UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')

def person_list():
    try:
        sqliteConnection = sqlite3.connect('database/userfr_fatigueanal.db', timeout=1000)
 
        print("Connected to SQLite")

        sqlite_select_query = """SELECT * from user_data"""

        df = pd.read_sql(sqlite_select_query, sqliteConnection)
        df['name'] = df['user_name'].replace('_', ' ', regex=True).str.title()
        person_list = df.values.tolist()
        
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("The Sqlite connection is closed")

    return render_template('person_list.html', person_list=person_list)

def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) # save image into upload folder
        # get predictions
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        
        # generate report
        report = []
        for i , obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image (array)
            eigen_image = obj['eig_img'].reshape(100,100) # eigen image (array)
            gender_name = obj['prediction_name'] # name 
            score = round(obj['score']*100,2) # probability score
            
            # save grayscale and eigne in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
            
            # save report 
            report.append([gray_image_name,
                           eig_image_name,
                           gender_name,
                           score])
            
        
        return render_template('gender.html',fileupload=True,report=report) # POST REQUEST
            
    
    
    return render_template('gender.html',fileupload=False) # GET REQUEST