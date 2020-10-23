from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle

app = Flask(__name__)
pickle_in = open('model-faketweet.pickle','rb')
pac = pickle.load(pickle_in)
tfid = open('tfid.pickle','rb')
tfidf_vectorizer = pickle.load(tfid)

@app.route('/')
def home():
 	return render_template("index.html")

@app.route('/newscheck')
def newscheck():	
	abc = request.args.get('news')	
	input_data = [abc.rstrip()]
	# transforming input
	tfidf_test = tfidf_vectorizer.transform(input_data)
	# predicting the input
	y_pred = pac.predict(tfidf_test)
	if y_pred==1:
		y_pred = "REAL"
	else:
		y_pred = "FAKE"
	return jsonify(result = y_pred)


if __name__=='__main__':
    app.run(debug=True)
