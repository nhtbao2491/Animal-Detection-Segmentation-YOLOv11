from ultralytics import YOLO
import os

def train_model():
    model = YOLO("yolo11m.pt")
    results = model.train(data="config.yaml", epochs=100, batch=8)

def predict_img(dir_model):
    name_model = dir_model.split("/")[-1]
    print(f"---------------------------- {name_model} ----------------------------")
    model = YOLO(dir_model)
    model.predict(source='./Detect_Img/Input', 
                    save=True,
                    conf=0.5,
                    project='./Detect_Img/Output',
                    name='Results',
                    save_txt=False,
                    save_crop=False)
    
def ls_models_dir():
    List_models_dir = "./List_models"
    os.makedirs(List_models_dir, exist_ok=True)
    model_paths = []
    for filename in os.listdir(List_models_dir):
        full_path = os.path.join(List_models_dir, filename)
        if os.path.isfile(full_path):
            normalized_path = full_path.replace('\\', '/')
            model_paths.append(normalized_path)
    return model_paths

if __name__ == "__main__":
    ds_dir_models = ls_models_dir()
    for dir in ds_dir_models:
        predict_img(dir)

    

