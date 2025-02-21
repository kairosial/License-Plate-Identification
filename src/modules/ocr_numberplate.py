import os
from paddleocr import PaddleOCR

def ocr_numberplate(image_path: str) -> str:
    """
    번호판 이미지에서 텍스트를 인식하는 함수.
    :param image_path: 번호판 이미지 파일 경로
    :return: 인식된 텍스트 문자열
    """
    rec_model_dir = r"src\utils\ocr_model"
    ocr = PaddleOCR(det=False, rec=True, rec_model_dir=rec_model_dir, lang="korean", use_gpu=True)
    
    result = ocr.ocr(image_path, cls=False)
    
    # 인식된 텍스트들을 하나의 문자열로 합쳐서 반환
    recognized_text = "".join([word_info[1][0] for line in result for word_info in line])
    return recognized_text
