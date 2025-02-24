import os
from datetime import datetime
from PIL import Image, ImageFile
import numpy as np
from ultralytics import YOLO

# 손상된 이미지도 로드할 수 있도록 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True

def detect_numberplate_bbox(image_path, model, resize=True, detection_save_dir=None):
    """
    원본 이미지를 열고, 옵션에 따라 640×640으로 리사이즈한 후 YOLO 모델로 번호판 영역을 detect합니다.
    동시에, YOLO의 detection 결과(바운딩박스가 그려진 이미지)를 지정한 detection_save_dir에 저장합니다.
    
    Parameters:
        image_path (str): 원본 이미지 파일 경로.
        model (YOLO): 미리 생성된 YOLO 모델 객체.
        resize (bool): True이면 원본 이미지를 640×640으로 리사이즈하여 detection 수행.
        detection_save_dir (str): 바운딩박스가 그려진 detection 결과 이미지를 저장할 디렉토리.
            기본값은 현재 파일 위치를 기준으로 output/detection/images 경로로 설정됩니다.
        
    Returns:
        tuple: (bboxes, (scale_x, scale_y), original_img)
            - bboxes: detection된 bounding box 좌표 배열 (xyxy 형식, float).
            - (scale_x, scale_y): detection 이미지 좌표를 원본 이미지로 변환하기 위한 스케일.
            - original_img: 원본 이미지 객체 (PIL.Image).
        만약 detection 실패 시 None을 반환합니다.
    """
    # 기본 detection_save_dir 설정
    if detection_save_dir is None:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        detection_save_dir = os.path.join(CURRENT_DIR, "..", "..", "..", "output", "detection", "images")
        detection_save_dir = os.path.abspath(detection_save_dir)
    
    try:
        original_img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    original_width, original_height = original_img.size

    if resize:
        detection_img = original_img.resize((640, 640), Image.LANCZOS)
        scale_x = original_width / 640
        scale_y = original_height / 640
    else:
        detection_img = original_img.copy()
        scale_x, scale_y = 1, 1

    detection_np = np.array(detection_img)
    results = model(detection_np, save=False)

    # detection 결과 이미지(바운딩박스가 그려진 이미지) 저장
    annotated_img_np = results[0].plot()
    annotated_img = Image.fromarray(annotated_img_np)
    os.makedirs(detection_save_dir, exist_ok=True)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    detection_filename = f"{image_basename}_detection.jpg"
    detection_filepath = os.path.join(detection_save_dir, detection_filename)
    annotated_img.save(detection_filepath)
    print(f"Detection result saved at: {detection_filepath}")

    # bounding box 좌표 추출 (xyxy 형식)
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No numberplate detected in the image.")
        return None

    try:
        bboxes = boxes.xyxy
        if hasattr(bboxes, "cpu"):
            bboxes = bboxes.cpu().numpy()
        else:
            bboxes = np.array(bboxes)
    except Exception as e:
        print(f"Error extracting bounding boxes: {e}")
        return None

    return bboxes, (scale_x, scale_y), original_img



def crop_numberplate_from_bbox(original_img, bboxes, scale, output_folder, image_basename):
    """
    원본 이미지와 detection된 bounding box, 스케일 정보를 받아서 번호판 영역을 crop하고 저장합니다.
    output_folder 내부에 'yymmdd_HHMMSS' 형식의 서브디렉토리를 생성한 후, 그 안에 crop된 이미지들을 저장합니다.
    
    Parameters:
        original_img (PIL.Image): 원본 이미지 객체.
        bboxes (ndarray): detection된 bounding box 좌표 배열 (xyxy 형식).
        scale (tuple): (scale_x, scale_y) 값.
        output_folder (str): crop된 이미지들이 저장될 기본 폴더 (예: cropped_numberplates).
        image_basename (str): 원본 이미지 파일명(확장자 제외), 저장 파일명에 사용.
    
    Returns:
        list: crop된 이미지 파일 경로들의 리스트.
    """
    scale_x, scale_y = scale
    original_width, original_height = original_img.size

    # output_folder 내부에 'yymmdd_HHMMSS' 형식의 서브디렉토리 생성
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    sub_output_folder = os.path.join(output_folder, timestamp)
    os.makedirs(sub_output_folder, exist_ok=True)

    cropped_files = []
    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = box
        orig_x1 = int(x1 * scale_x)
        orig_y1 = int(y1 * scale_y)
        orig_x2 = int(x2 * scale_x)
        orig_y2 = int(y2 * scale_y)

        orig_x1 = max(0, orig_x1)
        orig_y1 = max(0, orig_y1)
        orig_x2 = min(original_width, orig_x2)
        orig_y2 = min(original_height, orig_y2)

        cropped_img = original_img.crop((orig_x1, orig_y1, orig_x2, orig_y2))
        crop_filename = f"{image_basename}_crop_{idx}.jpg"
        crop_filepath = os.path.join(sub_output_folder, crop_filename)
        cropped_img.save(crop_filepath)
        print(f"Cropped image saved: {crop_filepath}")
        cropped_files.append(crop_filepath)

    return cropped_files



if __name__ == "__main__":
    # 모듈 테스트용 최소 메시지 출력
    print("crop_numberplate.py 모듈이 로드되었습니다. 이 모듈은 detect_numberplate_bbox와 crop_numberplate_from_bbox 함수를 제공합니다.")