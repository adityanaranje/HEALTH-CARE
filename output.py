import numpy as np
import pickle

def model1(dt):  
    try:
        to_predict_list = list(map(float, dt))
        lst = np.array(to_predict_list).reshape(1, 11)
        
        loaded_model = pickle.load(open("Models/cirrhosis.pkl", "rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1
    return result

def model2(dt):
    try:
        to_predict_list = list(map(float, dt))

        lst = np.array(to_predict_list).reshape(1, 10)

        ls1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float")

        ls1 = np.array(ls1).reshape(1, 15)
        ls1[0][0] = float(lst[0][0])
        ls1[0][1] = int(lst[0][1])
        ls1[0][2] = int(lst[0][2])
        ls1[0][3] = float(lst[0][3])
        ls1[0][4] = float(lst[0][4])
        ls1[0][5] = int(lst[0][5])
        ls1[0][6] = int(lst[0][6])
    
        if int(lst[0][7])==1:
            ls1[0][7]=1
        if int(lst[0][7])==2:
            ls1[0][8]=1
        if int(lst[0][7])==3:
            lst[0][9]=1
        if int(lst[0][7])==4:
            lst[0][10]=1

        ls1[0][11] = int(lst[0][8])

        if int(lst[0][9])==1:
            ls1[0][12]=1
        if int(lst[0][9])==2:
            ls1[0][13]=1
        if int(lst[0][9])==3:
            ls1[0][14]=1

        ls1 = np.array(ls1).reshape(1, 15)

        scaler = pickle.load(open("Scalers/scaler2.pkl","rb"))
    
        ls1 = scaler.transform(ls1) 
       
        loaded_model = pickle.load(open("Models/stroke.pkl", "rb"))
        result = loaded_model.predict(ls1)
    except:
        result = -1
    return result


def model3(dt):
    try:
        to_predict_list = list(map(float, dt))
        lst = np.array(to_predict_list).reshape(1,11)
        loaded_model = pickle.load(open("Models/heart.pkl","rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1
    return result

def model4(dt):  
    try:
        to_predict_list = list(map(float, dt))
        lst = np.array(to_predict_list).reshape(1, 9)
        loaded_model = pickle.load(open("Models/breast_cancer.pkl", "rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1
    return result

def model5(dt):  
    try:
        to_predict_list = list(map(float, dt))
        lst = np.array(to_predict_list).reshape(1, 12)
        loaded_model = pickle.load(open("Models/chronic_kidney.pkl", "rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1
    return result

def model6(dt):
    try:
        to_predict_list = list(map(float, dt))
        lst = np.array(to_predict_list).reshape(1,8)
        loaded_model = pickle.load(open("Models/diabetis.pkl","rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1
    
    return result


def model7(dt):
    try:
        to_predict_list = list(map(int, dt))
        lst = np.array(to_predict_list).reshape(1, 15)
        loaded_model = pickle.load(open("Models/lung_cancer.pkl", "rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1

    return result

def model8(dt):  
    try:
        to_predict_list = list(map(float, dt))
        lst = np.array(to_predict_list).reshape(1, 9)
        loaded_model = pickle.load(open("Models/water_potability.pkl", "rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1
    return result