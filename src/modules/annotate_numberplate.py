import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def annotate_from_detections(image, recognized_text, crop_results, output_folder: str = os.path.join('output', 'OCR'), font_size: int = 32):
    """
    검출된 번호판 영역 정보를 이용하여 원본 이미지에 바운딩 박스와 OCR 인식 결과(번호판 텍스트)를 표시한 주석 이미지를 생성합니다.
    
    :param image: 원본 이미지 (PIL Image 객체 또는 NumPy 배열)
    :param recognized_text: OCR 인식 결과 문자열
    :param crop_results: crop_numberplate 함수에서 반환한 리스트(각 항목에 "box" 키 포함)
    :param output_folder: 주석 이미지 저장 폴더
    :param font_size: 텍스트 폰트 크기
    :return: 주석 이미지(PIL 객체)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # NumPy 배열을 PIL Image로 변환
    if isinstance(image, np.ndarray):
        if image.shape[2] == 3:
            # 임의로 BGR로 가정하거나, 평균 채널값 등을 이용해 BGR/RGB 구분 로직을 짤 수도 있음
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    else:
        # PIL이라면 그냥 사용 (이미 RGB 모드가 맞는지 확인해볼 수도 있음)
        if image.mode != "RGB":
            image = image.convert("RGB")
    
    # 원본 이미지를 복사하여 주석 작업
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # 폰트 설정 (OS에 따라 경로 조정)
    try:
        if os.name == 'nt':
            font_path = "malgun.ttf"  # Windows의 경우
        else:
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
        print("지정 폰트를 로딩할 수 없어 기본 폰트를 사용합니다.")
    
    # 각 검출 영역에 대해 바운딩 박스와 텍스트 출력
    for res in crop_results:
        left, top, width, height = res["box"]
        box = (left, top, left + width, top + height)
        draw.rectangle(box, outline="red", width=4)
        # 텍스트 위치: 바운딩 박스 위쪽 또는 내부
        text_y = top - font_size if top - font_size > 0 else top
        draw.text((left, text_y), recognized_text, font=font, fill="yellow")
    
    return annotated_image