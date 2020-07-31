# Breast_Cancer_Detection
Using Random Forest and LightGBM, built a Breast Cancel Detection model and RESTful API where a user can upload the data and get results.

## Description
You work for the data team at a local research hospital. You've been tasked with developing a
means to help doctors diagnose breast cancer. You've been given data about biopsied breast
cells; where it is benign (not harmful) or malignant (cancerous).

## Data
***breast-cancer-wisconsin.txt***

**Columns**
```
Name                      Range or Description

Sample code number             id number
Clump Thickness                 1 - 10
Uniformity of Cell Size         1 - 10
Uniformity of Cell Shape        1 - 10
Marginal Adhesion               1 - 10
Single Epithelial Cell Size     1 - 10
Bare Nuclei                     1 - 10
Bland Chromatin                 1 - 10
Normal Nucleoli                 1 - 10
Mitoses                         1 - 10
Class                (2 for benign, 4 for malignant)
```

## RESTful Flask API
Pretrained LightGBM weights are saved in ```lgb.pkl```.

For a RESTful Flask API, run:
```
python app.py
```

A user can upload test data as in the same format as breast-cancer-wisconsin.txt and get the results.
