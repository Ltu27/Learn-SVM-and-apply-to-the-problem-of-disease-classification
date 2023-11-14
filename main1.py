import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Đọc file Excel
data = pd.read_excel('Exasens.xlsx')

# Tách dữ liệu đầu vào (features) và nhãn (labels)
X = data.drop('Diagnosis', axis=1)
X = np.array(X)

# Dữ liệu training chưa khử NaN bỏ id và nhãn
data_init = X[2:, 1:-4]
n = data_init.shape[0]
m = data_init.shape[1]

# Tạo dữ liệu khử NaN
data_new = []
for i in data_init:
    for j in i:
        j = str(j)
        if(j == 'nan'):
            data_new.append(0)
        else:
            data_new.append(j)

# Reshape dữ liệu thành ma trận n hàng m cột
data_finally = np.reshape(data_new, (n, m))

# Chuyển kiểu dữ liệu thành float để nhận data cuối cùng
data_finally = data_finally.astype(float)

# Label encoding nhãn, split 2 dòng đầu không có giá trị
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Diagnosis'])[2:]

# Chia dữ liệu thành bộ train và bộ test (test_size là phần trăm của bộ test)
X_train, X_test, y_train, y_test = train_test_split(data_finally, y, test_size=0.3, random_state=18)


# Hard Margin SVM
hvm_model = SVC(kernel='linear', C=1000)
hvm_model.fit(X_train, y_train)

yh_pred = hvm_model.predict(X_test)

h_accuracy = accuracy_score(y_test, yh_pred)
print("Độ chính xác trên bộ test:", h_accuracy)
a = []
for i in X_test:
    predicted_label = hvm_model.predict([i])
    decoded_label = label_encoder.inverse_transform(predicted_label)
    print("Nhãn dự đoán:", decoded_label)
    a.append(decoded_label)

str_a = str(a)
with open('HVM.txt', 'w') as f:
    f.write(str_a)


print("-----------------------------------------------------------------------------")
# Soft Margin SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác trên bộ test:", accuracy)

b = []
for i in X_test:
    predicted_label = hvm_model.predict([i])
    decoded_label = label_encoder.inverse_transform(predicted_label)
    print("Nhãn dự đoán:", decoded_label)
    b.append(decoded_label)

str_b = str(b)
with open('SVM.txt', 'w') as f:
    f.write(str_b)
