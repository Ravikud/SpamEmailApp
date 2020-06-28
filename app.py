import numpy as np
import pandas as pd
import pickle
from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df=pd.read_csv("YoutubeSpamMergedData.csv")
	df_data = df[["CONTENT","CLASS"]]
	df_x = df_data["CONTENT"]
	df_y = df_data.CLASS

	corpus = df_x
	cv = CountVectorizer()
	x = cv.fit_transform(corpus)
	from sklearn.model_selection import train_test_split
	x_train,x_test,y_train,y_test = train_test_split(x,df_x,test_size=0.25,random_state=42)
	
	#MultiNaive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	
	clf = MultinomialNB()
	clf.fit(x_train,y_train)
	clf.score(x_test,y_test)
	#clf = pickle.load(open("spam_detector_nb_model.pkl","rb"))
	ytb_model = open("spam_detector_nb_model.pkl","rb")
	clf = joblib.load(ytb_model)
	
	if request.method == 'POST':
		comment = request.form['comment']
		data=[comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('results.html',prediction = my_prediction)


if __name__ == "__main__":
    app.run()