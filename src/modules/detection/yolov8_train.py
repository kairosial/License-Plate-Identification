import os
from ultralytics import YOLO

def train_yolo_model(data_yaml, model_path, output_dir, epochs=20, imgsz=640, device="0"):
    # 결과 저장 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모델 불러오기 (pretrained weight 사용)
    model = YOLO(model_path)
    
    # 모델 학습: data_yaml 파일에 train/val 경로, 클래스 정보 등이 정의되어 있어야 함
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=output_dir  # 학습 결과가 저장될 디렉토리 지정 (옵션)
    )
    
    # 학습이 끝난 후 최종 모델은 model 객체에 남아있음.
    return model

if __name__ == "__main__":
    # 현재 스크립트 위치 기준 경로 설정
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(CURRENT_DIR, '..', '..', '..', 'output', 'detection', 'model')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # data.yaml 파일 경로 (Roboflow에서 받은 YAML 파일)
    data_yaml = '/home/azureuser/cloudfiles/code/LicensePlateDetection-3/data.yaml'
    # 사용할 pretrained 모델 (예: yolov8m.pt)
    model_path = 'yolov8m.pt'
    
    # 모델 학습 실행
    trained_model = train_yolo_model(data_yaml, model_path, output_dir, epochs=50, imgsz=640, device="0")
    
    # 필요하다면 학습 후 모델 저장 (예: ONNX 또는 .pt 파일 형식)
    trained_model.export(format="onnx")  # 예시: ONNX로 내보내기