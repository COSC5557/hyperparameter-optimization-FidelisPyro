
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# Path to my csv file
path = "/mnt/c/Users/kylel/Programming/School/PracticalML/wine+quality/winequality-red.csv"

# Read csv file into a pandas dataframe
df = pd.read_csv(path, delimiter = ";")

"""
# Check to see if there are any missing values
missing_values = df.isnull().sum()
print(missing_values)
"""


# Create a new dataframe without the quality column
quality = 'quality'
X = df.drop(quality, axis=1)
y = df['quality']

# Split the dataset into a training set and a temporary set into 60/40 split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the temporary set into validation and test set into 20/20 split of total dataset
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Only using numeric features because the wine dataset doesn't have any categorical features
numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())])

# Define the pipelines
pipelines = {
    'rf': Pipeline([('num_transformers', numeric_transformer), 
                    ('selector',RFECV(estimator = RandomForestClassifier(), step = 1, cv = 5)),
                    ('model', RandomForestClassifier())]),
    
    'svc': Pipeline([('num_transformers', numeric_transformer), 
                     ('selector',RFECV(estimator = SVC(kernel = 'linear'), step = 1, cv = 5)),
                     ('model', SVC())]),
    
    'nn': Pipeline([('num_transformers', numeric_transformer),  
                    ('model', MLPClassifier(max_iter=100000))]),
    
    'knn': Pipeline([('num_transformers', numeric_transformer), 
                     ('model', KNeighborsClassifier())]),
}



# Search for the best hyperparameters
# Separate search spaces for each model. One model uses only it's one search space
param_grids = {
    'rf': {
        'num_transformers__imputer__strategy': ['mean', 'median', 'most_frequent'],
        'num_transformers__scaler': [StandardScaler(), MinMaxScaler()],
        #'selector__min_features_to_select': [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'model__n_estimators': [50, 100],
        'model__max_depth': [5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 4, 8],
    },
    'svc': {
        'num_transformers__imputer__strategy': ['mean', 'median', 'most_frequent'],
        'num_transformers__scaler': [StandardScaler(), MinMaxScaler()],
        #'selector__min_features_to_select': [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'model__C': [0.001, 0.01, 0.1, 1, 10],
        'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'model__gamma': ['scale', 'auto'],
    },
    'nn': {
        'num_transformers__imputer__strategy': ['mean', 'median', 'most_frequent', 'constant'],
        'num_transformers__scaler': [StandardScaler(), MinMaxScaler()],
        'model__hidden_layer_sizes': [(50,), (100,)],
        'model__solver': ['lbfgs'],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
    },
    'knn': {
        'num_transformers__imputer__strategy': ['mean', 'median', 'most_frequent', 'constant'],
        'num_transformers__scaler': [StandardScaler(), MinMaxScaler()],
        'model__n_neighbors': [3, 5],
        'model__weights': ['uniform', 'distance'],
        'model__algorithm': ['brute'],
        'model__p': [1, 2],
    },
}


# Fit the pipelines and evaluate on val set
scores = {}
fitted_models = {}
num_features = {}
cv_results = {}

for model_name, pipeline in pipelines.items():
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    score = best_model.score(X_val, y_val)
    scores[model_name] = score  
    fitted_models[model_name] = best_model
    cv_results[model_name] = grid_search.cv_results_
    
    # Predict the validation set results
    y_pred = best_model.predict(X_val)
    
    # Compute confusion matrix
    confusion = confusion_matrix(y_val, y_pred)
    
    sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(f'Confusion_matrix_{model_name}.png')
    plt.close()  
    
    # Check if 'selector' step exists in the pipeline
    if 'selector' in best_model.named_steps:
        num_features[model_name] = best_model.named_steps['selector'].support_.sum()
    else:
        num_features[model_name] = X_train.shape[1]
        
best_rf_model = fitted_models['rf']
best_svc_model = fitted_models['svc']
best_nn_model = fitted_models['nn']
best_knn_model = fitted_models['knn']

best_rf_score = scores['rf']
best_svc_score = scores['svc']
best_nn_score = scores['nn']
best_knn_score = scores['knn']

print(f"Best RF model accuracy: {best_rf_score}")
print(f"Best SVC model accuracy: {best_svc_score}")
print(f"Best NN model accuracy: {best_nn_score}")
print(f"Best KNN model accuracy: {best_knn_score}")


for model_name, score in scores.items():
    print(f"{model_name}: {score}")

# Select best model
best_name, best_score = max(cv_results.items(), key=lambda x: np.mean(x[1]['mean_test_score']))
best_model = fitted_models[best_name]

# Predict the test set results
y_pred_best = best_model.predict(X_test)

best_confusion = confusion_matrix(y_test, y_pred_best)

def confustion_matrix_plot(model_name, confusion):
    sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Confusion Matrix " + model_name)
    plt.savefig(f'{model_name}_Confusion_matrix.png')
    plt.close()

def plot_cv(model_name, selector):
    # Plot the scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
    plt.title(f'{model_name.upper()}: number of features selected vs cross validation score')
    plt.savefig(f'{model_name}_pipeline_testing.png')
    plt.close()


def plot_mean_test_score(model_name, results):
    # Plot the scores
    plt.figure()
    plt.xlabel("Hyperparameter settings")
    plt.ylabel("Mean test score")
    plt.plot(range(1, len(results['mean_test_score']) + 1), results['mean_test_score'])
    plt.title(f'{model_name.upper()}: mean test score vs hyperparameter settings')
    plt.savefig(f'{model_name}_mean_test_score.png')
    plt.close()
    
def plot_score_features(scores, num_features):
    plt.figure()
    for model_name in scores.keys():
        plt.plot(num_features[model_name], scores[model_name], 'o', label = model_name)
    plt.xlabel("Number of features selected")
    plt.ylabel("Validation score")
    plt.legend()
    plt.grid(True)
    plt.title('Model Performance based on Number of Features and Best Hyperparameters')
    plt.savefig("Model_Performance.png")
    plt.close()

models_with_selection = ['rf', 'svc']

for model_name, pipeline in pipelines.items():
    if model_name in models_with_selection:

        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        selector = pipeline.named_steps['selector']

        plot_cv(model_name, selector)



# Best model evaluation on test set
test_score = best_model.score(X_test, y_test)
print(f"Best model: {best_name} with test score: {test_score}")
sns.heatmap(best_confusion, annot=True, cmap="BuPu", fmt="d")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Confusion Matrix Best" + best_name + "4")
plt.savefig('Confusion_matrix_Best' + best_name + '4.png')
plt.close()  


# Plot mean test score for each hyperparameter setting for each model
for model_name, results in cv_results.items():
    plot_mean_test_score(model_name, results)

plot_score_features(scores, num_features)



