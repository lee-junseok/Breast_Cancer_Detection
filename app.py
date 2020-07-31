from flask import Flask, request, render_template, flash,redirect, url_for, jsonify, make_response
from flask_restful import reqparse, abort, Api, Resource
import pickle
from werkzeug.utils import secure_filename
import werkzeug
import os
# from model import NLPModel
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# for modeling
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix



app = Flask(__name__)
api = Api(app)

class PredictBreastCancer(Resource):
    def post(self):
        df = pd.read_csv(request.files.get('file'))
        pred, predProba= self.model_prediction(df)
        output = pd.DataFrame({'Prediction':pred, "Score":predProba}).to_json(orient="records")
        return output

    def get(self):
        return make_response(render_template('upload.html'))



    def model_prediction(self,df):
        df['Class'] = df['Class'].apply(lambda x: 0 if x=='2' else (1 if x=='4' else 2)).astype('int8')
        target = 'Class'
        df = df.drop(columns = 'Index')
        df = pd.concat([df[['ID','Class']],df.drop(columns =['ID','Class'] )],axis = 1)
        df = df[df['Class']!=2]
        cols = list(df.columns)
        for col in cols[3:]:
            df[col] = df[col].apply(lambda x: int(x) if x.isnumeric() else None)
        df.fillna(df.mean(),inplace = True)
        for col in cols[1:]:
            df[col] = df[col].astype('int8')
        lgb_clf = joblib.load('lgb.pkl')

        model = lgb_clf
        predProba = model.predict_proba(df.drop(columns="Class"))
        # Make predictions on the validation data
        pred = model.predict(df.drop(columns="Class"))
        # auc_score = roc_auc_score(valY, predProba[:,1])
        return pred, predProba[:,1]
# Setup the Api resource routing here
# Route the URL to the resource

api.add_resource(PredictBreastCancer, '/')

if __name__ == '__main__':
    app.run(debug=True)

