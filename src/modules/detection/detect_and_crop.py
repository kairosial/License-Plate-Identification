import os
import numpy as np
import cv2
from PIL import Image, ImageFile
from ultralytics import YOLO

# 손상된 이미지를 로드 가능하도록 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True

def detect_numberplates_in_image(image, model, resize=True):
    """
    메모리 내 이미지(PIL 또는 numpy 배열)와 YOLO 모델을 입력받아,
    번호판 바운딩 박스(xyxy 형식)만 추출해 반환합니다.
    
    Parameters:
    -----------
    image : PIL.Image 또는 numpy.ndarray
        - PIL 이미지인 경우 RGB 모드 (또는 RGBA, P -> 내부에서 변환).
        - numpy 배열인 경우, 일반적으로 (H, W, C) 형식.
          OpenCV 형식(BGR)이면 BGR->RGB 변환.
    model : YOLO
        - 사전에 로드된 ultralytics YOLO 모델 객체
    resize : bool
        - True면 내부적으로 640x640 리사이즈 후 감지.
        - False면 원본 크기로 감지.
    
    Returns:
    --------
    bboxes : numpy.ndarray
        - shape: (N, 4), 각 행은 (x1, y1, x2, y2) float 좌표 (xyxy 형식).
        - 감지된 바운딩 박스가 없으면 빈 배열(길이 0).
    
    예외 상황:
    - 이미지가 손상되었거나, 모델 감지가 실패하면 빈 배열 반환.
    """

    # 1) PIL / numpy 배열인지 식별 및 RGB 보장
    pil_img = None
    if isinstance(image, Image.Image):
        # PIL 이미지인 경우 모드 확인
        if image.mode in ("RGBA", "P"):
            pil_img = image.convert("RGB")
        else:
            pil_img = image
    elif isinstance(image, np.ndarray):
        # numpy 배열인 경우 BGR -> RGB (OpenCV 형식이라 가정)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 간단히 BGR->RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
        else:
            # 혹은 이미 RGB라면 그대로 PIL 변환
            pil_img = Image.fromarray(image)
    else:
        print("detect_numberplates_in_image: 지원하지 않는 이미지 타입입니다.")
        return np.array([])  # 빈 배열

    # 2) 리사이즈 여부 결정
    original_width, original_height = pil_img.size
    if resize:
        resized_img = pil_img.resize((640, 640), Image.LANCZOS)
    else:
        resized_img = pil_img

    # 3) numpy 배열 변환 후 YOLO 추론
    detection_np = np.array(resized_img)
    results = model(detection_np, save=False)

    # 4) 바운딩 박스 추출
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No numberplate detected in the image.")
        return np.array([])  # 빈 배열

    bboxes = boxes.xyxy  # tensor or ndarray
    if hasattr(bboxes, "cpu"):
        bboxes = bboxes.cpu().numpy()
    else:
        bboxes = np.array(bboxes)

    # 5) 리사이즈로 인한 좌표 환산
    #    만약 "원본 좌표"가 필요하다면 아래 주석 해제:
    if resize:
        scale_x = original_width / 640
        scale_y = original_height / 640
        # bboxes[:, 0 or 2] => x1, x2
        # bboxes[:, 1 or 3] => y1, y2
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y

    return bboxes