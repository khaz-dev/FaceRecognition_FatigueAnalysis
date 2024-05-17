# Face Recognition Web App

This face recognition web app is integrated with machine learning model. Here user can upload an image and our machine learning model will analysis your Fatigue level based on data on user_data table in SQLite database.

- **Project URL**:  <https://khaz.one/fatigue_analysis/> (Deployed)

1. Desain Database (Done)
   I design database table to store user data with column :  Person ID, Person, Name, GFender, Age, Occupation, Sleep Duration, Sleep Quality, Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps and Sleep Disorder. You can see detail data on Person Tab.
3. Pengumpulan Dataset(Done)
4. Face Recognition(Done)
5. Fatigue Analysis(Done)
6. Integrasi Sistem(Done)
7. Deployment(Done)
   Deployment Aways Error, may be because face_recognition and opencv package, even i got some problem  when want to implement Deep Learning using it without conda.
My problem i only can deploy on Railway App because it's free, I can deploy it on Server or Cloud server, but I have not enough money and my trial has gone, please understanding. [So now i fix it by using cloud server]

Sorry if i take longer time because i try using all Algorithm Possible,
and mostly using deep learning hope to get better result but it take long time and much resource

For Fatigue Analysis, i using Up to Date Sleep Health and Lifestyle Analysis from Kaggle <https://www.kaggle.com/code/giulianoverdone/up-to-date-sleep-health-and-lifestyle-analysis/input/>, Because in the data still no label so I do Unsupervised learning to create Cluster then for the cluster i determine which cluster have Fit to Fatigue level, using K-Prototypes to mix Numerical and Categorical data to do clustering, and for Face Recognation i user Deep Learning package face_recognition. 

Data from Face Recognation i use hollywood artis foto, you can see all from this github. Take long to me to encoding Person Face Data (actually artist) and it take hours. For Fatigue Analysis I Using Logistic Regression then boosting the Classifier result with HistGradientBoostingClassifier.
you can se the proses and result on Project Output.

## Deployment Process (Local)
Yout need Python version 3.9 to run this
1. Create environment, on project directory open command prompt
```python
python -m venv myenv
```
2. Activate the virtual environment
```python
.\myenv\Scripts\activate
```
3. Install Packages
```python
pip install -r requirements.txt
```
4. After all the packages installed. Run the code
```python
python main.py
```

Then local server running and run the Web App.
Usually on Local (http://127.0.0.1:5000)



## Project Preview
![image](https://github.com/khaz-dev/facerec_fatiganal_app/blob/main/preview/preview_1.png)
![image](https://github.com/khaz-dev/facerec_fatiganal_app/blob/main/preview/preview_2.png)
![image](https://github.com/khaz-dev/facerec_fatiganal_app/blob/main/preview/preview_3.png)



### Deployment
![image](https://user-images.githubusercontent.com/75901421/184639715-7b4ba26c-6fb8-4157-8819-233b06dedb77.png)
