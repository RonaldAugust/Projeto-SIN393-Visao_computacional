import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from funcoes import segment_image, calculate_hu_moments, calculate_contour_signature, calculate_features

ds_path = 'C:\\Users\\Ronald\\Documents\\projeto_inicial_visao\\dataset'

classes_list = os.listdir(ds_path)

image_list = []
label_list = []
filename_list_ = []

for classe in classes_list:
    filename_list = os.listdir(os.path.join(ds_path, classe))
    for filename in filename_list:
        img_temp = plt.imread(os.path.join(ds_path, classe, filename))
        img_temp = transform.resize(img_temp, (img_temp.shape[0]//4, img_temp.shape[1]//4), anti_aliasing=True)
        segmented_img = segment_image(img_temp)
        image_list.append(segmented_img)
        label_list.append(classe)
        filename_list_.append(filename)

_, _, label_list_idx = np.unique(label_list, return_index=True, return_inverse=True)

image_list_temp = []
filename_list_temp = []

for i in range(6):
    image_list_temp += [image_list[j] for j in np.where(label_list_idx == i)[0][:6]]
    filename_list_temp += [filename_list_[j] for j in np.where(label_list_idx == i)[0][:6]]

fig, ax = plt.subplots(6, 6, figsize=(9, 5))

for i, (image, filename) in enumerate(zip(image_list_temp, filename_list_temp)):
    ax[i // 6, i % 6].imshow(image, vmin=0, vmax=1)
    ax[i // 6, i % 6].set_title(str(filename))
    ax[i // 6, i % 6].axis('off')

fig.tight_layout()
plt.show()

data = []
for img, label in zip(image_list, label_list):
    hu = calculate_hu_moments(img)
    signature = calculate_contour_signature(img)
    geom_features = calculate_features(img)
    features = {
        'label': label,
        **{f'hu_{i+1}': h for i, h in enumerate(hu)},
        'signature_mean': np.mean(signature),
        'signature_std': np.std(signature),
        **geom_features
    }
    data.append(features)

features_df = pd.DataFrame(data)
features_df.to_csv('caracteristicas.csv', index=False)
print("Características salvas com sucesso!")

X_train, X_test, y_train, y_test = train_test_split(features_df.drop('label', axis=1), 
                                                    features_df['label'], 
                                                    test_size=0.3, 
                                                    random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, 
                                                  random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("=== Modelo KNN ===")
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Acurácia: {accuracy_knn:.2f}")
print("Matriz de Confusão (KNN):")
print(confusion_matrix(y_test, y_pred_knn))
print("Relatório de Classificação (KNN):")
print(classification_report(y_test, y_pred_knn))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n=== Modelo Árvores de Decisão ===")
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Acurácia: {accuracy_dt:.2f}")
print("Matriz de Confusão (Árvores de Decisão):")
print(confusion_matrix(y_test, y_pred_dt))
print("Relatório de Classificação (Árvores de Decisão):")
print(classification_report(y_test, y_pred_dt))
