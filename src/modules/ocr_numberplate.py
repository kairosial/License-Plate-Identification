import os
import numpy as np
from paddleocr import PaddleOCR

def ocr_numberplate(image):
    """
    이미지(파일 경로가 아니라 PIL 이미지 또는 numpy 배열)를 받아서 OCR을 수행합니다.
    """
    # PaddleOCR 객체 생성 (det_model_dir은 유효한 경로 사용)
    os.makedirs(os.path.join('src', 'dummy_det'), exist_ok=True)
    rec_model_dir = os.path.join('src', 'utils', 'ocr_model')
    
    ocr = PaddleOCR(det=False,
                    det_model_dir=os.path.join('src', 'dummy_det'), 
                    rec=True,
                    rec_model_dir=rec_model_dir,
                    lang="korean",
                    use_gpu=True)
    
    # image가 numpy 배열이 아니라면 변환
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    result = ocr.ocr(image, cls=True)
    
    if not result or None in result:
        print("OCR 결과가 없습니다.")
        return ""
    
    try:
        recognized_text = "".join([word_info[1][0] for line in result for word_info in line if word_info and word_info[1]])
    except Exception as e:
        print("OCR 결과 처리 중 오류 발생:", e)
        recognized_text = ""
    
    return recognized_text