import pandas as pd
import os
import re
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import fetch_openml

# Expand DataFrame output to show all columns so there isnt any missing data that gets printed
pd.set_option('display.max_columns', None)

# --- MNIST Portion of the project ---
print("\n--- MNIST CLASSIFICATION ACCURACY ---")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mnist_results = []

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mnist_results.append(["Decision Tree", accuracy_score(y_test, y_pred)])

# Bagging Tree
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
mnist_results.append(["Bagging", accuracy_score(y_test, y_pred)])

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mnist_results.append(["Random Forest", accuracy_score(y_test, y_pred)])

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
mnist_results.append(["Gradient Boosting", accuracy_score(y_test, y_pred)])

mnist_df = pd.DataFrame(mnist_results, columns=["Classifier", "Accuracy"])
print(mnist_df)



#THE COMMENTED OUT CODE WAS RESPONSIBLE FOR F1 SCORE AND ACCURACY FROM DATA THIS WAS USED TO SOLVE THE FIRST PORTION OF THE PROJECT

#
# # Define the file path to obtain data
# data_dir = "/Users/heman/PycharmProjects/pythonProject2TreeBased/project2_data/all_data"
#
# # Collect all CSV file paths to extract data
# all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
#
# # Organize files by clause and dataset type for optimal utilization and to prevent misuse
# file_groups = defaultdict(dict)
# pattern = re.compile(r'(\w+)_c(\d+)_d(\d+)\.csv')
#
# for file_path in all_files:
#     match = pattern.search(os.path.basename(file_path))
#     if match:
#         dataset_type, clauses, samples = match.groups()
#         key = f"c{clauses}_d{samples}"
#         file_groups[key][dataset_type] = file_path
#
# # Filter only proper datasets (train, valid, test present)
# complete_datasets = {k: v for k, v in file_groups.items() if {'train', 'valid', 'test'}.issubset(v)}
#
# # Hyperparams for trees
# param_grid_tree = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': [5, 10, 20, None]
# }
#
# param_grid_bagging = {
#     'n_estimators': [10, 25, 50],
#     'max_samples': [0.5, 0.75, 1.0],
#     'estimator__max_depth': [5, 10, 20, None]
# }
#
# param_grid_rf = {
#     'n_estimators': [10, 50, 100],
#     'max_depth': [5, 10, 20, None],
#     'criterion': ['gini', 'entropy']
# }
#
# param_grid_gb = {
#     'n_estimators': [50, 100],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 10]
# }
#
# results_tree = []
# results_bagging = []
# results_rf = []
# results_gb = []
#
# for key, paths in complete_datasets.items():
#     # Load data
#     train_df = pd.read_csv(paths['train'], header=None)
#     valid_df = pd.read_csv(paths['valid'], header=None)
#     test_df = pd.read_csv(paths['test'], header=None)
#
#     # Split data X and y
#     X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
#     X_valid, y_valid = valid_df.iloc[:, :-1], valid_df.iloc[:, -1]
#     X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
#
#     # Combine all data for training and validation for final retraining
#     X_combined = pd.concat([X_train, X_valid], axis=0)
#     y_combined = pd.concat([y_train, y_valid], axis=0)
#
#     # Decision Tree Classifier
#     grid_search_tree = GridSearchCV(
#         DecisionTreeClassifier(random_state=42),
#         param_grid_tree,
#         cv=3,
#         scoring='f1',
#         n_jobs=-1
#     )
#     grid_search_tree.fit(X_train, y_train)
#     final_model_tree = DecisionTreeClassifier(**grid_search_tree.best_params_, random_state=42)
#     final_model_tree.fit(X_combined, y_combined)
#     y_pred_tree = final_model_tree.predict(X_test)
#     results_tree.append({
#         'Dataset': key,
#         'Accuracy': accuracy_score(y_test, y_pred_tree),
#         'F1 Score': f1_score(y_test, y_pred_tree)
#     })
#
#     # Bagging with Decision Trees
#     base_tree = DecisionTreeClassifier(random_state=42)
#     bagging_model = BaggingClassifier(estimator=base_tree, random_state=42)
#     grid_search_bagging = GridSearchCV(
#         bagging_model,
#         param_grid_bagging,
#         cv=3,
#         scoring='f1',
#         n_jobs=-1
#     )
#     grid_search_bagging.fit(X_train, y_train)
#     best_params_bagging = grid_search_bagging.best_params_
#     final_bagging = BaggingClassifier(
#         estimator=DecisionTreeClassifier(max_depth=best_params_bagging['estimator__max_depth'], random_state=42),
#         n_estimators=best_params_bagging['n_estimators'],
#         max_samples=best_params_bagging['max_samples'],
#         random_state=42
#     )
#     final_bagging.fit(X_combined, y_combined)
#     y_pred_bagging = final_bagging.predict(X_test)
#     results_bagging.append({
#         'Dataset': key,
#         'Accuracy': accuracy_score(y_test, y_pred_bagging),
#         'F1 Score': f1_score(y_test, y_pred_bagging)
#     })
#
#     # Random Forest Classifier
#     grid_search_rf = GridSearchCV(
#         RandomForestClassifier(random_state=42),
#         param_grid_rf,
#         cv=3,
#         scoring='f1',
#         n_jobs=-1
#     )
#     grid_search_rf.fit(X_train, y_train)
#     final_rf = RandomForestClassifier(**grid_search_rf.best_params_, random_state=42)
#     final_rf.fit(X_combined, y_combined)
#     y_pred_rf = final_rf.predict(X_test)
#     results_rf.append({
#         'Dataset': key,
#         'Accuracy': accuracy_score(y_test, y_pred_rf),
#         'F1 Score': f1_score(y_test, y_pred_rf)
#     })
#
#     # Gradient Boosting Classifier
#     grid_search_gb = GridSearchCV(
#         GradientBoostingClassifier(random_state=42),
#         param_grid_gb,
#         cv=3,
#         scoring='f1',
#         n_jobs=-1
#     )
#     grid_search_gb.fit(X_train, y_train)
#     final_gb = GradientBoostingClassifier(**grid_search_gb.best_params_, random_state=42)
#     final_gb.fit(X_combined, y_combined)
#     y_pred_gb = final_gb.predict(X_test)
#     results_gb.append({
#         'Dataset': key,
#         'Accuracy': accuracy_score(y_test, y_pred_gb),
#         'F1 Score': f1_score(y_test, y_pred_gb)
#     })
#
# # Convert and display all results
# results_df_tree = pd.DataFrame(results_tree)
# results_df_bagging = pd.DataFrame(results_bagging)
# results_df_rf = pd.DataFrame(results_rf)
# results_df_gb = pd.DataFrame(results_gb)
#
# print("Decision Tree Results:")
# print(results_df_tree[['Dataset', 'Accuracy', 'F1 Score']])
# print("\nBagging with Decision Trees Results:")
# print(results_df_bagging[['Dataset', 'Accuracy', 'F1 Score']])
# print("\nRandom Forest Results:")
# print(results_df_rf[['Dataset', 'Accuracy', 'F1 Score']])
# print("\nGradient Boosting Results:")
# print(results_df_gb[['Dataset', 'Accuracy', 'F1 Score']])
