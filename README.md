
# Nhận dạng Chữ số Viết tay bằng Mạng Nơ-ron Tích chập (CNN)

## 1. Giới thiệu

Dự án này nhằm xây dựng một mô hình nhận dạng chữ số viết tay sử dụng Mạng Nơ-ron Tích chập (CNN) với TensorFlow và Keras.  
Nhận dạng chữ số viết tay là bài toán kinh điển trong lĩnh vực thị giác máy tính, đòi hỏi mô hình phải xử lý các ảnh xám có kích thước 28×28 với nhiều kiểu chữ khác nhau.  
Mục tiêu của dự án là:
- Số hóa và tiền xử lý dữ liệu MNIST.
- Xây dựng và huấn luyện mô hình CNN với các lớp Convolution, MaxPooling, Flatten, Dense và Dropout.
- Đánh giá mô hình trên tập validation và test.
- Lưu lại mô hình đã huấn luyện.
- Hiển thị một số dự đoán mẫu và trực quan hóa kết quả (biểu đồ loss, accuracy, ma trận nhầm lẫn).

**Kết quả đạt được**:  
- Độ chính xác trên tập kiểm tra đạt khoảng 99.94%  
- Loss đạt giá trị khoảng 4.5% (hoặc theo thông số của mô hình huấn luyện)

## 2. Chuẩn bị dữ liệu và Cài đặt Thư viện

- **Dữ liệu**:  
  Bộ dữ liệu MNIST bao gồm 42.000 mẫu ảnh huấn luyện và 28.000 mẫu ảnh test (đối với bài toán dự thi, chỉ sử dụng tập train và tập test có nhãn).  
  Mỗi ảnh là ảnh xám với kích thước 28×28 pixel.

- **Cài đặt thư viện**:  
  Các thư viện chính sử dụng là:  
  - TensorFlow, Keras  
  - Pandas, NumPy  
  - Matplotlib, Seaborn

Cài đặt bằng lệnh:
```bash
pip install tensorflow pandas numpy matplotlib seaborn
```

## 3. Tiền xử lý Dữ liệu

- **Load dữ liệu**:  
  Dữ liệu được đọc từ file CSV (ví dụ: `train.csv` và `test.csv`) trên Google Drive thông qua Google Colab.
  
- **Khám phá dữ liệu**:  
  - Kiểm tra thông tin dữ liệu (`df.info()`, `df.shape`, `df.head()`)
  - Kiểm tra dữ liệu bị thiếu bằng heatmap và hàm `isna()`.
  
- **Tiền xử lý**:
  - Tách các pixel (từ cột 1 đến 785) thành dữ liệu đầu vào X và nhãn chữ số từ cột 0 thành y.
  - Chuẩn hóa giá trị pixel (chia cho 255) để chuyển đổi về khoảng [0, 1].
  - Chuyển đổi hình dạng dữ liệu từ dạng vector thành ma trận 28×28 và thêm kênh màu (channel) = 1.

## 4. Trực quan hóa Dữ liệu

- **Phân phối nhãn**:  
  Vẽ biểu đồ cột thể hiện số lượng mẫu của từng chữ số (0-9).

- **Giảm chiều bằng T-SNE**:  
  Lấy mẫu một phần dữ liệu (ví dụ 5000 mẫu) và sử dụng thuật toán T-SNE để trực quan hóa dữ liệu trên mặt phẳng 2D, giúp nhận diện sự phân bố và cụm của các chữ số.

- **Hiển thị hình ảnh mẫu**:  
  Vẽ lưới các ảnh chữ số viết tay cùng với nhãn để quan sát cách chữ số được viết.

## 5. Xây dựng và Huấn luyện Mô hình CNN

Mô hình được xây dựng theo kiến trúc:
- **Convolution2D**: Lớp tích chập với kernel_size=5, filters=8 cho lớp đầu, filters=16 cho lớp thứ hai.
- **MaxPooling2D**: Giảm kích thước ảnh sau mỗi lớp tích chập.
- **Flatten**: Chuyển đổi đầu ra từ các lớp tích chập thành vector.
- **Dense**: Lớp fully-connected với 128 đơn vị và hàm kích hoạt ReLU.
- **Dropout**: Giảm overfitting với tỷ lệ dropout 20%.
- **Dense (Softmax)**: Lớp đầu ra với 10 đơn vị và hàm kích hoạt softmax để biểu diễn xác suất cho từng chữ số.

Mô hình được biên dịch với:
- **Optimizer**: Adam với learning rate = 0.001.
- **Loss function**: `sparse_categorical_crossentropy` (vì nhãn chỉ là các chỉ số số nguyên).
- **Metrics**: Accuracy.

### Huấn luyện Mô hình
Mô hình được huấn luyện trong 10 epoch, sử dụng tensorboard callback để theo dõi quá trình học. Đồ thị loss và accuracy được vẽ ra để kiểm tra quá trình huấn luyện và đánh giá trên tập validation.

## 6. Đánh giá Mô hình

- **Đánh giá trên tập training và validation**:
  - Tính toán loss và accuracy.
  - Vẽ đồ thị đường cong loss và accuracy theo epoch.
  
- **Dự đoán và Trực quan hóa**:
  - Hiển thị một số dự đoán trên tập validation (hiển thị ảnh kèm dự đoán và nhãn thực).
  - Vẽ ma trận nhầm lẫn của tập validation để đánh giá mức độ nhận dạng đúng của mô hình.
  - Tính các chỉ số: Accuracy, Precision, Recall, F1-score dựa trên ma trận nhầm lẫn.

## 7. Lưu và Tải Mô hình

Sau khi huấn luyện, mô hình được lưu dưới dạng file H5 (`digits_recognition_cnn.h5`) và có thể tải lại để dự đoán sau này.  
Ngoài ra, một số thông tin về dự đoán trên tập test cũng được in ra.

## 8. Hướng phát triển

- **Tăng cường dữ liệu (Data Augmentation)**: Áp dụng thêm các biến đổi ảnh để tăng số lượng mẫu huấn luyện.
- **Tinh chỉnh kiến trúc mô hình**: Thử nghiệm với các kiến trúc CNN phức tạp hơn (ví dụ: VGG, ResNet) để cải thiện độ chính xác.
- **Triển khai mô hình**: Tích hợp mô hình vào ứng dụng thực tế hoặc website để nhận dạng chữ số viết tay từ ảnh người dùng tải lên.

## 9. Cách chạy Dự án

1. **Mount Google Drive** (nếu sử dụng Google Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Chạy Notebook**:  
   Mở file notebook (ví dụ: `handwritten_digit_recognition.ipynb`) trên Google Colab hoặc Jupyter Notebook và chạy các cell theo thứ tự từ tiền xử lý, trực quan hóa, huấn luyện cho đến đánh giá.
3. **Theo dõi Kết quả**:  
   - Xem đồ thị loss, accuracy.
   - Kiểm tra ma trận nhầm lẫn và các chỉ số đánh giá.
   - Xem các ảnh dự đoán để đối chiếu với nhãn thật.

## 10. Yêu cầu Cài đặt
- Python 3.7+
- Các thư viện: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## 11. Tài liệu Tham khảo
- [TensorFlow Documentation](https://www.tensorflow.org/guide/keras)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras Documentation](https://keras.io/)

## 12. Liên hệ
Nếu có thắc mắc hoặc góp ý, vui lòng mở issue trên GitHub hoặc liên hệ qua email: thanh.van19062004@gmail.com
```

---

### Hướng dẫn sử dụng README này

1. Tạo file `README.md` trong repository của bạn.
2. Sao chép và dán nội dung trên vào file.
3. Điều chỉnh thông tin như tên file, email, đường dẫn đến dữ liệu, số epoch, hoặc các thông số huấn luyện khác cho phù hợp với dự án của bạn.
4. Commit và push file `README.md` lên GitHub.

