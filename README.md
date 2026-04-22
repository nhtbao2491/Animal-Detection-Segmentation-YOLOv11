# Animal Detection & Segmentation với YOLOv11

Dự án nhận diện và phân đoạn động vật sử dụng mô hình **YOLOv11** (Detection + Segmentation), được huấn luyện trên tập dữ liệu **Animals-151** từ Kaggle. Hỗ trợ nhận diện **13 loài động vật** phổ biến trong tự nhiên.

---

## Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Các loài động vật hỗ trợ](#-các-loài-động-vật-hỗ-trợ)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
- [Huấn luyện mô hình](#-huấn-luyện-mô-hình)
- [Đánh giá mô hình](#-đánh-giá-mô-hình)
- [Dự đoán (Inference)](#-dự-đoán-inference)
- [Mô tả các file chính](#-mô-tả-các-file-chính)

---

## Giới thiệu

Dự án này xây dựng pipeline hoàn chỉnh từ thu thập dữ liệu, tiền xử lý, huấn luyện đến suy luận (inference) cho bài toán **nhận diện và phân đoạn động vật** trong ảnh.

- **Task**: Object Detection + Instance Segmentation
- **Model**: YOLOv11m (Detection) & YOLOv11m-seg (Segmentation)
- **Dataset**: Animals-151 (Kaggle) — ảnh `.jpg`, kích thước `640×640 pixels`, 3 kênh màu RGB
- **Loại dữ liệu**: Unstructured Data, Supervised Learning, Classification/Detection

---

## 🐾 Các loài động vật hỗ trợ

| # | Tên loài | # | Tên loài |
|---|----------|---|----------|
| 0 | Beagle Dog | 7 | Pig |
| 1 | Blackbuck Antelope | 8 | Red Panda |
| 2 | Cat | 9 | Rhino |
| 3 | Cow | 10 | Snow Leopard |
| 4 | Elephant | 11 | Tiger |
| 5 | Lion | 12 | Zebra |
| 6 | Panda | | |

---

## Cấu trúc dự án

```
project/
│
├── Dataset/
│   ├── train/              # Ảnh huấn luyện (theo từng label)
│   │   ├── Beagle_Dog/
│   │   ├── Cat/
│   │   └── ...
│   └── val/                # Ảnh validation (5 ảnh/label)
│       ├── Beagle_Dog/
│       └── ...
│
├── Create_newLabel/
│   ├── Input_Img/          # Ảnh đầu vào để tạo label mới
│   └── Output_Folder/      # Ảnh đã được đặt tên và phân loại
│
├── Transform_Img/
│   ├── Input_Img/          # Ảnh cần resize/chuyển đổi định dạng
│   └── Output_Img/         # Ảnh sau khi đã xử lý (640×640, .jpg)
│
├── Input_Train/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── zip/
│
├── List_models/            # Thư mục chứa các model .pt để predict
│
├── Detect_Img/
│   ├── Input/              # Ảnh đầu vào để nhận diện
│   └── Output/Results/     # Kết quả nhận diện
│
├── runs/                   # Kết quả huấn luyện (tự động tạo)
│   ├── detect/train/weights/best.pt
│   └── segment/train/weights/best.pt
│
├── yolo11m.pt              # Pretrained weights Detection
├── yolo11m-seg.pt          # Pretrained weights Segmentation
├── config.yaml             # File cấu hình dataset cho YOLO
├── DS_Labels.txt           # Danh sách label
│
├── Processed.py            # Script tiền xử lý dữ liệu
├── yolov11.py              # Script huấn luyện và dự đoán
├── data_Img.ipynb          # Notebook phân tích tập dữ liệu
└── processed.ipynb         # Notebook xử lý & huấn luyện
```

---

## Yêu cầu hệ thống

- Python >= 3.9
- CUDA >= 11.8 (khuyến nghị để train trên GPU)
- RAM >= 8GB
- GPU VRAM >= 6GB (khuyến nghị khi train)

---

## Cài đặt

**1. Clone repository**

```bash
git clone https://github.com/nhtbao2491/Animal-Detection-Segmentation-YOLOv11.git
cd Animal-Detection-Segmentation-YOLOv11
```

**2. Tạo môi trường ảo (khuyến nghị)**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

**3. Cài đặt các thư viện cần thiết**

```bash
pip install ultralytics
pip install Pillow
pip install opencv-python
pip install matplotlib
```

## Chuẩn bị dữ liệu

### Tải dataset

Tải tập dữ liệu **Animals-151** từ Kaggle:  
🔗 [https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset)

Giải nén và đặt vào thư mục `./Dataset/`.

---

### Xử lý dữ liệu bằng `Processed.py`

Khởi tạo đối tượng `YoloPreprocessor`:

```python
from Processed import YoloPreprocessor
processor = YoloPreprocessor()
```

#### 1. Tạo label mới từ ảnh thô

Đặt ảnh vào `./Create_newLabel/Input_Img/`, sau đó chạy:

```python
processor.create_newLabel("Rhino")
# Ảnh sẽ được đặt lại tên thành Rhino_1.jpg, Rhino_2.jpg, ...
# và chuyển vào ./Create_newLabel/Output_Folder/Rhino/
```

#### 2. Resize & chuyển đổi định dạng ảnh

Đặt ảnh vào `./Transform_Img/Input_Img/`, sau đó chạy:

```python
processor.transform_Image()
# Ảnh được resize về 224×224, chuyển sang .jpg
# Kết quả lưu tại ./Transform_Img/Output_Img/
```

>  **Lưu ý:** Khi huấn luyện YOLO, ảnh cần kích thước `640×640`. Chỉnh lại `img.resize((640, 640))` trong hàm nếu cần.

#### 3. Chia dữ liệu train/val

```python
processor.split_dataset()
# Tự động chia: 5 ảnh/label → val, còn lại → train
# Kết quả: ./Dataset/train/ và ./Dataset/val/
```

#### 4. Tạo file DS_Labels.txt

```python
processor.create_DSLabels()
# Tạo danh sách tên label từ ./Dataset/train/
# Lưu vào DS_Labels.txt
```

#### 5. Chuẩn bị cấu trúc thư mục đầu vào để train

```python
YoloPreprocessor.prepare_inputTraining()
# Copy ảnh vào ./Input_Train/images/train & val
```

---

## Huấn luyện mô hình

### Cấu hình `config.yaml`

Tạo file `config.yaml` với nội dung mẫu:

```yaml
path: ./Input_Train
train: images/train
val: images/val

nc: 13
names:
  - Beagle_Dog
  - Blackbuck_Antelope
  - Cat
  - Cow
  - Elephant
  - Lion
  - Panda
  - Pig
  - Red_Panda
  - Rhino
  - Snow_Leopard
  - Tiger
  - Zebra
```

### Chạy training

```python
from yolov11 import train_model
train_model()
```

Hoặc chạy trực tiếp:

```python
from ultralytics import YOLO

# Detection
model = YOLO("yolo11m.pt")
results = model.train(data="config.yaml", epochs=100, batch=8, imgsz=640, device=0)

# Segmentation
model_seg = YOLO("yolo11m-seg.pt")
results_seg = model_seg.train(data="config.yaml", epochs=100, batch=16, imgsz=640, device=0)
```

Kết quả được lưu tại:
- `./runs/detect/train/weights/best.pt` — model Detection tốt nhất
- `./runs/segment/train/weights/best.pt` — model Segmentation tốt nhất

---

## Đánh giá mô hình

```python
from ultralytics import YOLO

detect = YOLO('./runs/detect/train/weights/best.pt')
segment = YOLO('./runs/segment/train/weights/best.pt')

metrics_detect = detect.val(data='config.yaml', imgsz=640, batch=16)
metrics_segment = segment.val(data='config.yaml', imgsz=640, batch=16)
```

Các chỉ số đánh giá bao gồm:

| Chỉ số | Mô tả |
|--------|-------|
| **Precision** | Tỷ lệ dự đoán đúng trên tổng dự đoán |
| **Recall** | Tỷ lệ phát hiện đúng trên tổng thực tế |
| **F1-score** | Trung bình điều hòa Precision & Recall |
| **mAP@0.5** | Mean Average Precision tại IoU = 0.5 |
| **mAP@0.5:0.95** | Mean Average Precision trung bình nhiều ngưỡng IoU |

---

## Dự đoán (Inference)

### Dự đoán với một model cụ thể

```python
from yolov11 import predict_img

predict_img('./runs/detect/train/weights/best.pt')
# Ảnh đầu vào: ./Detect_Img/Input/
# Kết quả lưu tại: ./Detect_Img/Output/Results/
```

### Dự đoán hàng loạt với nhiều model

Đặt các file `.pt` vào `./List_models/`, sau đó chạy:

```bash
python yolov11.py
```

Script sẽ tự động lấy tất cả model trong thư mục và chạy predict tuần tự.

### Tham số predict

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `source` | `./Detect_Img/Input` | Thư mục ảnh đầu vào |
| `conf` | `0.5` | Ngưỡng confidence (0–1) |
| `save` | `True` | Lưu ảnh kết quả |
| `save_txt` | `False` | Không lưu nhãn dạng txt |
| `save_crop` | `False` | Không lưu vùng cắt |

---

## Mô tả các file chính

| File | Mô tả |
|------|-------|
| `Processed.py` | Class `YoloPreprocessor` — xử lý dữ liệu, tạo label, chia train/val |
| `yolov11.py` | Huấn luyện, predict đơn và predict hàng loạt |
| `data_Img.ipynb` | Notebook phân tích và trực quan hóa tập dữ liệu |
| `processed.ipynb` | Notebook pipeline xử lý dữ liệu và training |
| `yolo11m.pt` | Pretrained weights YOLOv11m (Detection) |
| `yolo11m-seg.pt` | Pretrained weights YOLOv11m-seg (Segmentation) |
| `DS_Labels.txt` | Danh sách 13 label động vật |
| `config.yaml` | Cấu hình dataset cho YOLO training |

---

## Ghi chú

- Khi thêm label mới, chạy lại `create_DSLabels()` và cập nhật `config.yaml`.
- Nên train trên GPU để rút ngắn thời gian (đặt `device=0`); nếu dùng CPU thì đặt `device='cpu'`.
- Tập val mặc định có **5 ảnh/label** — có thể điều chỉnh trong hàm `split_dataset()`.

---

## License & Usage

Dự án này sử dụng framework YOLO từ Ultralytics.

- YOLO (Ultralytics) được phát hành dưới giấy phép **AGPL-3.0**
- Điều này có nghĩa:
  - Bạn được phép sử dụng, chỉnh sửa và chia sẻ cho mục đích học tập và nghiên cứu
  - Nếu sử dụng trong sản phẩm thương mại hoặc hệ thống có backend (web/app), bạn **phải công khai mã nguồn theo AGPL-3.0**

### Lưu ý về YOLOv11
- “YOLOv11” trong dự án này có thể là phiên bản tùy chỉnh hoặc không chính thức
- Người dùng cần kiểm tra nguồn gốc và giấy phép trước khi sử dụng cho mục đích thương mại

### Model weights
- Các file `.pt` (model weights) trong repository chỉ phục vụ mục đích học tập
- Không đảm bảo phù hợp cho sử dụng thương mại nếu chưa kiểm tra rõ license

---

## Credits

- YOLO framework: Ultralytics  
- Dataset: Animals-151 từ Kaggle

---