import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Đường dẫn đến tệp Excel
path_to_excel_file = 'Exasens.xlsx'

# Đọc dữ liệu từ tệp Excel
data_frame = pd.read_excel(path_to_excel_file)
data_frame = np.array(data_frame)
data = np.zeros(([data_frame.shape[0], data_frame.shape[1]]))

# Dữ liệu training chưa khử NaN bỏ id và nhãn
data = data_frame[2:, 2:-4]
n = data.shape[0]
m = data.shape[1]

# Tạo dữ liệu khử NaN
data_new = []
for i in data:
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

# Nhãn
label = data_frame[2:, :1]
label = label.ravel()

# Bộ dữ liệu huấn luyện
X_train = data_finally

# Bộ nhãn huấn luyện
y_train = label

# Tách đặc trưng và nhãn
features = data_finally
labels = label

# Mã hóa nhãn thành các số nguyên
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


# Khởi tạo mô hình Soft SVM
svm_model = SVC(kernel='linear', C=1000.0)

# Huấn luyện mô hình
svm_model.fit(features, encoded_labels)

# Dự đoán nhãn cho một mẫu mới
new_sample = [0, 0,	0,	0,	0,	45,	3]
predicted_label = svm_model.predict([new_sample])

# Giải mã nhãn dự đoán
decoded_label = label_encoder.inverse_transform(predicted_label)

print("Nhãn dự đoán:", decoded_label)
