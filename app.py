from flask import Flask, render_template, request
import pickle   
import numpy as np
import output
import random
app = Flask(__name__)   # Flask constructor
  

# Main Page
@app.route('/') 
@app.route('/home')     
def home():

    quotes = ["It is health that is the real wealth, and not pieces of gold and silver",
    "The cheerful mind perseveres, and the strong mind hews its way through a thousand difficulties",
    "I have chosen to be happy because it is good for my health",
    "A sad soul can be just as lethal as a germ",
    "Remain calm, because peace equals power",
    "Healthy citizens are the greatest asset any country can have",
    "Motivation is what gets you started. Habit is what keeps you going",
    "The only bad workout is the one that didn't happen",
    "Challenging yourself every day is one of the most exciting ways to live",
    "When you feel like quitting, think about why you started",
    "The same voice that says 'give up' can also be trained to say 'keep going' "]

    get_quotes = random.sample(quotes, 3)


    return render_template('home.html', val1=get_quotes[0], val2=get_quotes[1], val3=get_quotes[2])    


# First Page  
@app.route('/cirrhosis',methods=['GET','POST'])
def cirrhosis():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = output.model1(to_predict_list)
        
        return render_template('cirrhosis.html',prediction=result)

    return render_template('cirrhosis.html')


# Second Page
@app.route('/stroke',methods=['GET','POST'])
def stroke():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()

        to_predict_list = list(to_predict_list.values())
        
        result = output.model2(to_predict_list)
        
        return render_template('stroke.html',prediction=result)

    return render_template('stroke.html')


# Third Page
@app.route('/heartdisease',methods=['GET','POST'])
def heartdisease():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = output.model3(to_predict_list)
        return render_template('heartdisease.html',prediction = result)
        
    return render_template('heartdisease.html')


# Fourth Page
@app.route('/breastcancer',methods=['GET','POST'])
def breastcancer():
    lst = [1,2,3,4,5,6,7,8,9,10]
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = output.model4(to_predict_list)
            
        return render_template('breastcancer.html',prediction=result, lst=lst)
    
    return render_template('breastcancer.html', lst=lst)


# Fifth Page
@app.route('/chronickidneydisease',methods=['GET','POST'])
def chronickidneydisease():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = output.model5(to_predict_list)
            
        return render_template('chronickidneydisease.html',prediction=result)
    
    return render_template('chronickidneydisease.html')


# Sixth Page
@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        result = output.model6(to_predict_list)
        return render_template('diabetes.html',prediction = result)
    return render_template('diabetes.html')


# Seventh Page
@app.route('/lungcancer',methods=['GET','POST'])
def lungcancer():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        result = output.model7(to_predict_list)

        return render_template('lungcancer.html',prediction=result)

    return render_template('lungcancer.html')

# Eighth Page
@app.route('/waterpotability',methods=['GET','POST'])
def waterpotability():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        result = output.model8(to_predict_list)
            
        return render_template('waterpotability.html',prediction=result)
        
    return render_template('waterpotability.html')

if __name__=='__main__':
   app.run(debug=False)
