# MOCKPROJECTVTI

## Tổng quan dự án

MOCKPROJECTVTI là một dự án Machine Learning dự đoán churn khách hàng, được triển khai dưới dạng ứng dụng Streamlit để hỗ trợ phân tích dữ liệu và đưa ra dự đoán theo thời gian thực.

Dự án bao gồm quy trình đầy đủ từ tạo dữ liệu, khám phá dữ liệu, xây dựng mô hình cho đến triển khai giao diện dự đoán. Người dùng có thể nhập thông tin khách hàng và nhận kết quả churn cùng xác suất dự đoán.

## Mục tiêu

- Dự đoán khả năng khách hàng rời bỏ dịch vụ.
- Hỗ trợ phân tích các yếu tố ảnh hưởng đến churn.
- Cung cấp giao diện trực quan để thử nghiệm mô hình.

## Thành phần chính

- Notebook 1: tạo dữ liệu giả lập hoặc xử lý dữ liệu đầu vào.
- Notebook 2: phân tích dữ liệu khám phá và trực quan hóa xu hướng.
- Notebook 3: xây dựng, đánh giá và lưu mô hình dự đoán.
- `app.py`: ứng dụng Streamlit phục vụ dự đoán churn và hiển thị dashboard.
- `models/best_model.joblib`: mô hình đã huấn luyện và ngưỡng dự đoán.
- `Data/churn_data_customer.csv`: dữ liệu tham chiếu dùng cho ứng dụng.

## Dữ liệu và đặc trưng

Ứng dụng sử dụng các nhóm thông tin chính như:

- Thông tin cơ bản của khách hàng.
- Mức độ sử dụng và tần suất hoạt động.
- Lịch sử tương tác, thanh toán và hỗ trợ.

Trong ứng dụng, một số đặc trưng được mã hóa và tạo thêm như:

- `subscription_type_enc`
- `has_active_promo_enc`
- `engagement_score`
- `satisfaction_score`
- `usage_trend_3m_6m`

## Cách hoạt động của ứng dụng

1. Tải mô hình đã huấn luyện từ thư mục `models/`.
2. Đọc dữ liệu tham chiếu từ thư mục `Data/`.
3. Người dùng nhập thông tin khách hàng trên giao diện.
4. Hệ thống tiền xử lý dữ liệu và căn chỉnh với các cột đặc trưng của mô hình.
5. Ứng dụng trả về xác suất churn và nhãn dự đoán.

## Công nghệ sử dụng

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- plotly
- joblib

## Cấu trúc thư mục

```text
mockprojectvti/
├── 1_Data_Generation.ipynb
├── 2_Exploratory_Data_Analysis.ipynb
├── 3_Model_Building_moi.ipynb
├── app.py
├── Data/
│   ├── churn_data_customer.csv
│   └── customer_churn_dataset_generated.csv
├── models/
│   └── best_model.joblib
├── requirements.txt
└── README.md
```

## Chạy dự án

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sau khi khởi động, mở giao diện Streamlit trong trình duyệt để nhập dữ liệu và xem kết quả dự đoán churn.
