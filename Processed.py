import os
import shutil
import random
from PIL import Image


class YoloPreprocessor:
    def create_newLabel(self , label_name):
        input_dir = './Create_newLabel/Input_Img'
        output_dir = './Create_newLabel/Output_Folder'
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        label_folder = os.path.join(output_dir, label_name)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_dir, image_file)
            new_name = f"{label_name}_{idx}.jpg"
            new_image_path = os.path.join(label_folder, new_name)
            shutil.move(image_path, new_image_path)
        print(f"Đã tạo thành công label : {label_name}")

    def split_dataset(self):
        source_dir = "./Create_newLabel/Output_Folder/"
        train_dir = "./Dataset/train"
        val_dir = "./Dataset/val"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        for label in os.listdir(source_dir):
            label_path = os.path.join(source_dir, label)
            if not os.path.isdir(label_path):
                continue
            train_label_path = os.path.join(train_dir, label)
            val_label_path = os.path.join(val_dir, label)
            if os.path.exists(train_label_path) or os.path.exists(val_label_path):
                continue
            images = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
            random.shuffle(images)
            val_images = images[:5]
            train_images = images[5:]
            train_label_path = os.path.join(train_dir, label)
            val_label_path = os.path.join(val_dir, label)
            os.makedirs(train_label_path, exist_ok=True)
            os.makedirs(val_label_path, exist_ok=True)
            for img in val_images:
                src = os.path.join(label_path, img)
                dst = os.path.join(val_label_path, img)
                shutil.copy(src, dst)
            for img in train_images:
                src = os.path.join(label_path, img)
                dst = os.path.join(train_label_path, img)
                shutil.copy(src, dst)
        print(f"Chia dữ liệu thành ----> Tran & Val thành công.")

    def transform_Image(self):
        input_folder = './Transform_Img/Input_Img'
        output_folder = './Transform_Img/Output_Img'
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpeg', '.jpg', '.bmp', '.webp')):
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((224, 224))
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + '.jpg')
                img_resized.save(output_path, format='JPEG', quality=95)
                os.remove(img_path)
        print("Hoàn tất chuyển đổi và resize ảnh.")

    def create_DSLabels(self):
        source_dir = "./Dataset/train"
        output_file = "DS_Labels.txt"
        label_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
        if not label_folders:
            print("Không có label trong thư mục './Dataset/train' để tạo danh sách.")
            return
        labels = sorted([label for label in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, label))])
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, label in enumerate(labels):
                f.write(f"{idx}: {label}\n")
        print(f"Đã ghi danh sách label vào {output_file}")

    def prepare_inputTraining():
        src_train = './Dataset/train'
        src_val = './Dataset/val'
        dst_images_train = './Input_Train/images/train'
        dst_images_val = './Input_Train/images/val'
        dst_labels_train = './Input_Train/labels/train'
        dst_labels_val = './Input_Train/labels/val'
        dst_zip_train = './Input_Train/zip/train'
        dst_zip_val = './Input_Train/zip/val'
        os.makedirs(dst_images_train, exist_ok=True)
        os.makedirs(dst_images_val, exist_ok=True)
        os.makedirs(dst_labels_train, exist_ok=True)
        os.makedirs(dst_labels_val, exist_ok=True)
        os.makedirs(dst_zip_train, exist_ok=True)
        os.makedirs(dst_zip_val, exist_ok=True)
        def copy_images(src_folder, dst_folder):
            for filename in os.listdir(src_folder):
                src_file = os.path.join(src_folder, filename)
                dst_file = os.path.join(dst_folder, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
        copy_images(src_train, dst_images_train)
        copy_images(src_val, dst_images_val)
        print("Đã hoàn tất tạo cấu trúc dữ liệu đầu vào để train model.")
        