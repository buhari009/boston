'''from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os;

# Monkey patching the SVC class to change its behavior
def new_decision_function(self, X):
    return -self._decision_function(X)

SVC._decision_function = new_decision_function

# UPLOAD_FOLDER = './uploads'
UPLOAD_FOLDER = './'
data_dir = settings.MEDIA_ROOT

def load_data(filename):
    data = pd.read_csv(os.path.join(data_dir, filename))
    return data

def split_data(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return round(acc, 2)*100, report

def index(request):
    return render(request, "index.html")

def upload(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        filename = file.name
        fs = FileSystemStorage(location=data_dir)
        fs.save(filename, file)
        data = load_data(filename)
        X_train, X_test, y_train, y_test = split_data(data)
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        model = train_model(X_train_scaled, y_train)
        acc, report = evaluate_model(model, X_test_scaled, y_test)
        return render(request, "result.html", context={
            'acc': acc,
            'precision_0': report["0"]["precision"],
            'recall_0': report["0"]["recall"],
            'f1_score_0': report["0"]["f1-score"],
            'support_0': report["0"]["support"],
            'precision_1': report["1"]["precision"],
            'recall_1': report["1"]["recall"],
            'f1_score_1': report["1"]["f1-score"],
            'support_1': report["1"]["support"]
        })
    return redirect('index')
'''

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

# Monkey patching the SVC class to change its behavior
def new_decision_function(self, X):
    return -self._decision_function(X)

SVC._decision_function = new_decision_function

# UPLOAD_FOLDER = './uploads'
UPLOAD_FOLDER = './'
data_dir = settings.MEDIA_ROOT

def load_data(filename):
    data = pd.read_csv(os.path.join(data_dir, filename))
    # drop any rows with missing values
    data = data.dropna()
    # remove any rows with a value of 0 for the Glucose, BloodPressure, SkinThickness, Insulin, or BMI columns
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    data = data[~(data[zero_cols] == 0).any(axis=1)]
    # replace any values of 0 for the Pregnancies or Age columns with the median value
    data["Pregnancies"] = data["Pregnancies"].replace(0, data["Pregnancies"].median())
    data["Age"] = data["Age"].replace(0, data["Age"].median())
    return data

def split_data(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return round(acc, 2)*100, report

def index(request):
    return render(request, "index.html")

def upload(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        filename = file.name
        fs = FileSystemStorage(location=data_dir)
        fs.save(filename, file)
        data = load_data(filename)
        X_train, X_test, y_train, y_test = split_data(data)
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        model = train_model(X_train_scaled, y_train)
        acc, report = evaluate_model(model, X_test_scaled, y_test)
        return render(request, "result.html", context={
            'acc': acc,
            'precision_0': report["0"]["precision"],
            'recall_0': report["0"]["recall"],
            'f1_score_0': report["0"]["f1-score"],
            'support_0': report["0"]["support"],
            'precision_1': report["1"]["precision"],
            'recall_1': report["1"]["recall"],
            'f1_score_1': report["1"]["f1-score"],
            'support_1': report["1"]["support"]
        })
    return redirect('index');            
           
