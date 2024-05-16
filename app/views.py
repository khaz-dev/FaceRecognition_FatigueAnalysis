import os
import cv2
from ml.face_recognitions import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as matimg
import sqlite3
import pandas as pd
import pickle


UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


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

def fatigue_analysis():
    fatigue_model = "model/fatigue_anal_model.pkl"

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
            user_name = obj['prediction_name'] # name 
            # score = round(obj['score']*100,2) # probability score
            
            # save grayscale and eigne in predict folder
            gray_image_name = f'face_roi_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image)


            # Load data from database
            # Search Data in Database base on username
            try:
                sqliteConnection = sqlite3.connect('database/userfr_fatigueanal.db', timeout=1000)

                print("Connected to SQLite")
                print('user_name : ', user_name)
                sqlite_select_query = f"""SELECT * from user_data WHERE user_name ='{user_name}'"""

                df = pd.read_sql(sqlite_select_query, sqliteConnection)
                data = df.drop(['user_id', 'user_name'], axis='columns')
                data.rename(columns={'gender': 'Gender', 'age': 'Age', 'occupation': 'Occupation', 'sleep_duration': 'Sleep Duration', 'quality_sleep': 'Quality of Sleep', 'physical_activity': 'Physical Activity Level', 'stress_level': 'Stress Level', 'bmi_category': 'BMI Category', 'blood_pressure': 'Blood Pressure', 'heart_rate': 'Heart Rate', 'daily_steps': 'Daily Steps', 'sleep_disorder': 'Sleep Disorder'}, inplace=True)

            except sqlite3.Error as error:
                print("Failed to read data from sqlite table", error)
            finally:
                if sqliteConnection:
                    sqliteConnection.close()
                    print("The Sqlite connection is closed")

            # Do Fatigue Analysis
            # See Data searched
            print('type data : ', type(data))
            print('data content : ', data)
            # Load Model
            if len(data) == 0 :
                print('No data to Fatigue Analysis')
                fatigue_level = [0]
            else :
                model = pickle.load(open(fatigue_model, 'rb'))
                print('Type Model : ', type(model))
                print('Model : ', model)
                print('################################################')
                fatigue_level = model.predict(data)
                print('Type Fatigue level : ', type(fatigue_level))
                print('Type Fatigue level : ', fatigue_level)
                fatigue_level = fatigue_level + 1


            
            user_name = user_name.replace('_',' ').title()

            # save report 
            report.append([gray_image_name,
                           user_name,
                           fatigue_level[0]])      
        
        return render_template('fatigue_analysis.html',fileupload=True,report=report) # POST REQUEST
    return render_template('fatigue_analysis.html',fileupload=False) # GET REQUEST