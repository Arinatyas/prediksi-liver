# -*- coding: utf-8 -*-
"""Artificial_UAS_Kelompok 10.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_o0Q9xyNWHpysrRQ9wngSjINE0RwSkMm

**IMPORT DATA**
"""
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/Arinatyas/prediksi-liver/refs/heads/main/indian_liver_patient.csv", encoding="utf-8")
# df

df.info()


df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
# df

df.isnull().sum()

columns_with_null = df.columns[df.isnull().any()].tolist()

for col in columns_with_null:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        print(f"Column '{col}' is not numeric and will not be processed for mean fill.")

# print(df.isnull().sum())

df['Dataset'] = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)
# df

df.duplicated()

df.drop_duplicates(inplace=True)

# df.duplicated()

# output_csv_file = 'cleaned_data.csv'
# df.to_csv(output_csv_file, index=False)

df = pd.read_csv('https://raw.githubusercontent.com/Arinatyas/prediksi-liver/refs/heads/main/cleaned_data%20(1).csv')
test_size = int(len(df) * 0.3)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = df.iloc[:-test_size, :-1].values
y_train = df.iloc[:-test_size, -1].values
X_test = df.iloc[-test_size:, :-1].values

pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False, header=["label"])
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)

print("Data X_train berhasil disimpan ke X_train.csv")
print("Data y_train berhasil disimpan ke y_train.csv")
print("Data X_test berhasil disimpan ke X_test.csv")


import numpy as np

class Node:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self, value):
        self.value = value

feature_names = df.columns[:-1]

def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def information_gain(y, left_y, right_y):
    p_left = len(left_y) / len(y)
    p_right = len(right_y) / len(y)
    return entropy(y) - p_left * entropy(left_y) - p_right * entropy(right_y)

ig_scores = {}

for idx, feature in enumerate(feature_names):
    values = np.unique(X_train[:, idx])
    ig_feature = 0

    for val in values:
        left_mask = X_train[:, idx] == val
        right_mask = ~left_mask
        left_y = y_train[left_mask]
        right_y = y_train[right_mask]

        ig = information_gain(y_train, left_y, right_y)

        ig_feature = max(ig_feature, ig)

    ig_scores[feature] = ig_feature

for feature, ig in ig_scores.items():
    print(f"Information Gain untuk {feature}: {ig:.4f}")

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth < self.max_depth and len(np.unique(y)) > 1:
            feat_idxs = np.random.choice(n_features, n_features, replace=False)
            best_feat, best_thresh = self._best_split(X, y, feat_idxs)
            if best_feat is not None:
                left_mask = X[:, best_feat] < best_thresh
                left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
                right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
                return Node(best_feat, best_thresh, left, right)

        return LeafNode(self._most_common_label(y))

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split = None
        for feat_idx in feat_idxs:
            thresholds, classes = zip(*sorted(zip(X[:, feat_idx], y)))
            for i in range(1, len(y)):
                left_y = classes[:i]
                right_y = classes[i:]
                gain = information_gain(y, left_y, right_y)
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    split = (feat_idx, thresholds[i])
        return split if split is not None else (None, None)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if isinstance(node, LeafNode):
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

import numpy as np

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = [DecisionTree(max_depth=self.max_depth) for _ in range(self.n_trees)]
        for tree in self.trees:
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._majority_vote(tree_pred) for tree_pred in tree_preds.T])

    def _majority_vote(self, tree_preds):
        unique, counts = np.unique(tree_preds, return_counts=True)
        return unique[np.argmax(counts)]

rf = RandomForest(n_trees=10, max_depth=5)
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)

train_accuracy = np.mean(y_train == y_train_pred)
print(f"Akurasi pada data train: {train_accuracy:.2f}")

def save_tree(tree, file):
    if isinstance(tree, DecisionTree):
        tree = tree.tree

    if isinstance(tree, LeafNode):
        return {'type': 'leaf', 'value': tree.value}
    else:
        left = save_tree(tree.left, file)
        right = save_tree(tree.right, file)
        return {'type': 'node', 'feature': tree.feature, 'threshold': tree.threshold, 'left': left, 'right': right}

with open('random_forest_model.txt', 'w') as f:
    for tree in rf.trees:
        tree_dict = save_tree(tree, f)
        f.write(f"{tree_dict}\n")

print("Model berhasil disimpan dalam 'random_forest_model.txt'")

def load_tree(tree_dict):
    if tree_dict['type'] == 'leaf':
        return LeafNode(tree_dict['value'])
    else:
        left = load_tree(tree_dict['left'])
        right = load_tree(tree_dict['right'])
        return Node(tree_dict['feature'], tree_dict['threshold'], left, right)

def load_trees_from_file(filename):
    trees = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tree_dict = eval(line.strip())
            tree = load_tree(tree_dict)
            trees.append(tree)
    return trees

loaded_trees = load_trees_from_file('random_forest_model.txt')
rf_loaded = RandomForest(n_trees=len(loaded_trees), max_depth=5)
rf_loaded.trees = [DecisionTree() for _ in range(len(loaded_trees))]
for i, tree in enumerate(loaded_trees):
    rf_loaded.trees[i].tree = tree

test_data = pd.read_csv('X_test.csv')

if isinstance(X_test, pd.DataFrame):
    X_test = X_test.to_numpy()

y_test_pred = rf_loaded.predict(X_test)

print("\nPrediksi pada data pengujian:")
for i in range(10):
    print(f"Data: {X_test[i]}, Prediksi: {y_test_pred[i]}")

X_test_with_predictions = np.column_stack((X_test, y_test_pred))
column_names = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio", "predicted"
]
# output_df = pd.DataFrame(X_test_with_predictions, columns=column_names)
# output_df.to_csv('hasil_test.csv', index=False)

import cloudpickle
import streamlit as st

# Simpan model menggunakan cloudpickle
with open('random_forest_model.pkl', 'wb') as f:
    cloudpickle.dump(rf, f)

def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        return cloudpickle.load(f)

# Memuat model hanya sekali
model = load_model()

# ✅ Prediction function using the loaded model
def predict_liver_disease(input_data):
    prediction = model.predict(input_data)
    return "Positive" if prediction[0] == 1 else "Negative"

# ✅ Fungsi untuk memprediksi penyakit hati
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                          alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                          albumin, albumin_and_globulin_ratio):
    input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                            albumin, albumin_and_globulin_ratio]])
    prediction = model.predict(input_data)
    return "Positive" if prediction[0] == 1 else "Negative"

# ✅ Aplikasi Streamlit
st.title('Prediksi Penyakit Hati')

# 🔹 Inputan dari pengguna
age = st.number_input('Masukkan umur (Age)', min_value=0)
gender = st.radio('Masukkan jenis kelamin (Gender)', [0, 1], format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
total_bilirubin = st.number_input('Masukkan Total Bilirubin')
direct_bilirubin = st.number_input('Masukkan Direct Bilirubin')
alkaline_phosphotase = st.number_input('Masukkan Alkaline Phosphotase')
alamine_aminotransferase = st.number_input('Masukkan Alamine Aminotransferase')
aspartate_aminotransferase = st.number_input('Masukkan Aspartate Aminotransferase')
total_protiens = st.number_input('Masukkan Total Proteins')
albumin = st.number_input('Masukkan Albumin')
albumin_and_globulin_ratio = st.number_input('Masukkan Albumin and Globulin Ratio')

# 🔹 Prediksi dilakukan di latar belakang
result = predict_liver_disease(input_data)

# 🔹 Tombol untuk menampilkan hasil prediksi
if st.button('Prediksi'):
    st.write(f'**Hasil Prediksi:** {result}')
