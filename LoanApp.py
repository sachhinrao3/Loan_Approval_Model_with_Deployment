from flask import Flask, request, render_template
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open("Loan_AN_LR_Model.pkl",'rb'))

@app.route('/')
def home():
    return render_template('index.html')


def LoanAN(FICO, Amount):
    out = model.predict_proba([[FICO, Amount]])[0][1]
    
    if out > 0.8:
        return ("#####Congrats your loan is approved########")
    
    else:
        return (".......Your loan is not approved..........(:")

@app.route("/predict", methods=['POST'])
def predict():
    '''
    For Rendering Results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features)
    prediction = LoanAN(final_features[0],final_features[1])
    
    return render_template('index.html', prediction_text = prediction)
    
if __name__ == "__main__":
    app.run(debug= True)
    
    
    

