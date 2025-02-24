import os
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import numpy as np

def detect_numberplate(model_path, image_path, output_dir, resize=True):
    """
    지정된 모델 파일을 로드한 후, image_path가 파일이면 단일 이미지에 대해,
    디렉토리면 해당 디렉토리 내의 모든 이미지 파일에 대해 객체 검출을 수행합니다.
    결과 이미지는 output_dir 내의 현재 날짜/시간 폴더에 입력 이미지 파일명으로 저장됩니다.
    
    Parameters:
        model_path (str): 학습된 모델 파일 경로
        image_path (str): 객체 검출을 진행할 이미지 파일 경로 또는 이미지들이 담긴 디렉토리 경로
        output_dir (str): 결과 이미지가 저장될 기본 디렉토리 경로 (하위에 날짜/시간 폴더가 생성됨)
        resize (bool): True인 경우 이미지를 640×640으로 리사이즈, False인 경우 원본 이미지 사용
    """
    # 모델 로드
    model = YOLO(model_path)
    
    # image_path가 파일인지 디렉토리인지 확인 후, 처리할 이미지 파일 리스트 생성
    if os.path.isfile(image_path):
        image_files = [image_path]
    elif os.path.isdir(image_path):
        allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            os.path.join(image_path, f) for f in os.listdir(image_path)
            if f.lower().endswith(allowed_exts)
        ]
    else:
        print(f"주어진 경로가 파일이나 디렉토리가 아닙니다: {image_path}")
        return
    
    results_dict = {}
    for file in image_files:
        # 이미지 열기
        image = Image.open(file)
        
        # resize 옵션에 따라 640×640으로 리사이즈
        if resize:
            image = image.resize((640, 640), Image.LANCZOS)
        
        # numpy 배열로 변환 (YOLO 모델은 numpy 배열 입력 지원)
        image_np = np.array(image)
        
        # 객체 검출 수행 (save=False로 자동 저장 기능 비활성화)
        results = model(image_np, save=False)
        
        # 첫 번째 결과 객체의 plot() 메서드를 통해 annotated 이미지 생성
        annotated_img = results[0].plot()
        
        # 입력 파일명으로 output_dir 내에 저장
        output_file = os.path.join(output_dir, os.path.basename(file))
        Image.fromarray(annotated_img).save(output_file)
        print(f"Detection result saved at: {output_file}")
        
        results_dict[file] = results
        
    return results_dict

if __name__ == "__main__":
    # 기본 output 디렉토리 설정
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(CURRENT_DIR, '..', '..', '..', 'output', 'detection', 'images')
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # 현재 날짜와 시간으로 서브 디렉토리 생성 ('yymmdd_hhmmss' 형식)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습된 모델 파일 경로 설정
    model_path = '/home/azureuser/cloudfiles/code/Users/6b011/LPR/License-Plate-Identification/output/detection/model/train/weights/best.pt'
    
    # 검출할 이미지 경로 (단일 이미지 파일 혹은 이미지들이 담긴 디렉토리)
    image_path = '/your/image/path'
    
    # resize 옵션을 True 또는 False로 설정 (기본값은 True)
    results = detect_numberplate(model_path, image_path, output_dir, resize=False)
