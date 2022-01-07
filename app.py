from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def result():
    Item_Weight = float(request.form['Item_Weight'])
    Item_Fat_Content = float(request.form['Item_Fat_Content'])
    Item_Visibility = float(request.form['Item_Visibility'])
    Item_Type = float(request.form['Item_Type'])
    Item_MRP = float(request.form['Item_MRP'])
    Outlet_Size = float(request.form['Outlet_Size'])
    Outlet_Location_Type = float(request.form['Outlet_Location_Type'])
    Outlet_Type = float(request.form['Outlet_Type'])
    Outlet_Years = float(request.form['Outlet_Years'])
    Item_Type_Combined = float(request.form['Item_Type_Combined'])


    X = np.array([[Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Size,Outlet_Location_Type,Outlet_Type,Outlet_Years,Item_Type_Combined ]])
    model = pickle.load(open('sales.pkl','rb'))
    y_predict = model.predict(X)
    return jsonify({'Prediction': float(y_predict)})

if __name__ == "__main__":
    app.run(debug=True,port=1234)
