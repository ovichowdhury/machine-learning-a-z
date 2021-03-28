from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler


dataset = pd.read_csv("Social_Network_Ads.csv")

print(dataset.shape)
print(dataset.head())

columns_to_encode = ['Gender']
columns_to_scale = ['Age', 'EstimatedSalary']

encoder = OneHotEncoder(sparse=False)
scaler = StandardScaler()

encoded_columns = encoder.fit_transform(dataset[columns_to_encode])
scaled_columns = scaler.fit_transform(dataset[columns_to_scale])

print("shape: ", encoded_columns.shape)

processed_dataset = np.concatenate([encoded_columns, scaled_columns], axis=1)

dataset = pd.concat(
    [pd.DataFrame(processed_dataset), dataset.Purchased], axis=1)

print(dataset.head())


X = dataset.iloc[:, :-1].values

Y = dataset.iloc[:, 4].values


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)


classifier = SVC(kernel='rbf', random_state=0)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

# Classification Report
print(classification_report(Y_test, Y_pred, target_names=["Not Purchased", "Purchased"]))


# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

print(cm)

disp = plot_confusion_matrix(classifier, X_test, Y_test,
                             display_labels=["Not Purchased", "Purchased"],
                             cmap=plt.cm.Blues,
                             normalize=None
                             )

disp.ax_.set_title("Confusion Matrix of Social Ads")
plt.show()
