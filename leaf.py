# Training and saving the model
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import pickle

categories = ['Healthy', 'Powdery Mildew', 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back']
image_size = (128, 128)

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

all_images = []
all_labels = []
for category in categories:
    category_path = os.path.join(r'C:\Users\pakcomp\Downloads\Mango_db\MangoLeafBD Dataset', category)
    images, labels = load_images_from_folder(category_path, category)
    all_images.extend(images)
    all_labels.extend(labels)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

label_encoder = LabelEncoder()
all_labels = label_encoder.fit_transform(all_labels)

n_samples, height, width, n_channels = all_images.shape
X = all_images.reshape(n_samples, -1)
y = all_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='poly', degree=3,C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

model_path = r'C:\Users\pakcomp\Downloads\Mango_db\MangoLeafBD Dataset\trained_plant_disease_model.pkl'
label_encoder_path = r'C:\Users\pakcomp\Downloads\Mango_db\MangoLeafBD Dataset\label_encoder.pkl'

with open(model_path, "wb") as model_file:
    pickle.dump(model, model_file)

with open(label_encoder_path, "wb") as le_file:
    pickle.dump(label_encoder, le_file)
