# Face Recognition Web App

This face recognition web app is integrated with machine learning model. Here user can upload an image and our machine learning model will detect face and predict gender (Male or Female).

- **Project URL**:  <https://khaz-facerec-fatigue-analysis.up.railway.app/>

1. Desain Database (Done)
2. Pengumpulan Dataset(Done)
3. Face Recognition(Done)
4. Fatigue Analysis(Done)
5. Integrasi Sistem(Done)
6. Deployment(Done)
   Deployment Aways Error, may be because face_recognition and opencv package, even i got some problem  when want to implement Deep Learning using it without conda.
My problem i only can deploy on Railway App because it's free, I can deploy it on Server or Cloud server, but I have not enough money and my trial has gone, please understanding.

Sorry if i take longer time because i try using all Algorithm Possible,
and mostly using deep learning hope to get better result but it take long time and much resource

For Fatigue Analysis, i using Up to Date Sleep Health and Lifestyle Analysis from Kaggle <https://www.kaggle.com/code/giulianoverdone/up-to-date-sleep-health-and-lifestyle-analysis/input/>, Because in the data still no label so I do Unsupervised learning to create Cluster then for the cluster i determine which cluster have Fit to Fatigue level, using K-Prototypes to mix Numerical and Categorical data to do clustering, and for Face Recognation i user Deep Learning package face_recognition and Data from Face Recognation i use hollywood artis foto, you can see al from this github. Take long to me to encoding Person Face Data (actually artist) and it take hours. Fatigue Analysis I Using Logistic Regression then boosting the result Classifier result usign HistGradientBoostingClassifier.
you can se the proses and result on Project Output.


### Project Output
![image](https://github.com/khaz-dev/facerec_fatiganal_app/blob/main/preview/preview_1.png)
![image](https://github.com/khaz-dev/facerec_fatiganal_app/blob/main/preview/preview_2.png)
![image](https://github.com/khaz-dev/facerec_fatiganal_app/blob/main/preview/preview_3.png)

### Deployment
![image](https://user-images.githubusercontent.com/75901421/184639715-7b4ba26c-6fb8-4157-8819-233b06dedb77.png)
