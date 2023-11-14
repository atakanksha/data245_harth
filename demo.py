import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class_code_to_class_name = {
    1: 'walking',
    2: 'running',
    3: 'shuffling',
    4: 'stairs (ascending)',
    5: 'stairs (descending)',
    6: 'standing',
    7: 'sitting',
    8: 'lying',
    13: 'cycling (sit)',
    14: 'cycling (stand)',
    130: 'cycling (sit, inactive)',
    140: 'cycling (stand, inactive)',
}

class_code_to_id = {}
id_to_class_code = {}
id_to_class_name = {}
for i, code in enumerate(class_code_to_class_name.keys()):
    class_code_to_id[code] = i
    id_to_class_code[i] = code
    id_to_class_name[i] = class_code_to_class_name[code]

def load_model(filename):
    with open(filename, 'rb') as file:  
        model = pickle.load(file)

    return model

def remap_labels(y):
    '''Converts class codes into consecutive ids.'''
    remap_y = y.copy()
    for class_code, class_id in class_code_to_id.items():
        remap_y[y == class_code] = class_id
    return remap_y

X = np.load('features.npy')
y_orig = np.load('labels.npy')
y = remap_labels(y_orig)

test_size = 0.2
train_size = 1 - test_size
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_seed)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

models = dict(
    knn_clf = load_model('./knn.pkl'),
    svm_clf = load_model('./svm.pkl'),
    logisitic_regression_clf = load_model('./lr.pkl'),
    random_forest_clf = load_model('./rf.pkl'),    
    xgboost_clf = load_model('./xg_boost.pkl')
)

names = list(id_to_class_name.values())

for name, model in models.items():
    print(f'Evaluating {name}')
    y_test_pred = model.predict(X_test_norm)
    report = classification_report(
        y_test, y_test_pred, target_names=names, digits=4
    )
    print(report)
    print('#'*50)