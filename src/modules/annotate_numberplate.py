import os
import platform
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def load_custom_font(font_size=50):
    """
    다양한 OS 환경에서 폰트를 유연하게 로드.
    Windows/macOS/Linux에서 사용 가능한 폰트 경로를 순차적으로 확인하고,
    적절한 폰트를 찾아서 반환합니다.
    만약 모든 폰트를 찾지 못하면 기본 폰트(ImageFont.load_default())를 사용합니다.
    """
    system_os = platform.system().lower()
    possible_font_paths = []

    # OS별로 자주 사용되는 폰트 경로들을 등록
    if "windows" in system_os:
        # Windows
        possible_font_paths = [
            r"C:\Windows\Fonts\malgun.ttf",       # 맑은 고딕
            r"C:\Windows\Fonts\malgunbd.ttf",     # 맑은 고딕 (볼드)
            r"C:\Windows\Fonts\gulim.ttf",        # 굴림
            r"C:\Windows\Fonts\msgothic.ttc",     # MS 고딕
        ]
    elif "darwin" in system_os:
        # macOS
        possible_font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/NanumGothic.ttf",
        ]
    else:
        # Linux (Ubuntu, CentOS, etc.)
        possible_font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
        ]

    # 순차적으로 시도
    for font_path in possible_font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                pass

    print("사용 가능한 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
    return ImageFont.load_default()


def annotate_from_detections(
    image,
    detections_info,
    output_folder: str = os.path.join('output', 'OCR'),
    font_size: int = 50
):
    """
    검출된 번호판 영역 정보를 이용하여 원본 이미지에 바운딩 박스와
    OCR 인식 결과(번호판 텍스트)를 표시한 주석 이미지를 생성합니다.
    
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
            # 임의로 BGR로 가정 -> RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    else:
        # PIL 이미지인 경우, RGB 모드 보장
        if image.mode != "RGB":
            image = image.convert("RGB")
    
    # 원본 이미지를 복사하여 주석 작업
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # 개선된 폰트 로더 사용
    font = load_custom_font(font_size)

    # 각 검출 영역에 대해 바운딩 박스와 텍스트 출력
    for info in detections_info:
        left, top, width, height = info["box"]
        recognized_text = info["text"]
        box = (left, top, left + width, top + height)
        draw.rectangle(box, outline="red", width=4)

        text_y = top - font_size if (top - font_size) > 0 else top
        draw.text((left, text_y), recognized_text, font=font, fill="yellow")
    
    return annotated_image