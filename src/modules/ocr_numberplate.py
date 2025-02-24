import os
from paddleocr import PaddleOCR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def ocr_numberplate(image_path: str):
    # PaddleOCR 객체 생성 (det_model_dir은 유효한 경로를 사용)
    rec_model_dir = os.path.join(CURRENT_DIR, '..', 'utils', 'ocr_model')

    ocr = PaddleOCR(det=False, rec=True, rec_model_dir=rec_model_dir, lang="korean", use_gpu=True)
    
    # OCR 수행
    result = ocr.ocr(image_path, cls=True)
    
    # 결과가 None 또는 빈 리스트인 경우 처리
    if not result or None in result:
        print("OCR 결과가 없습니다.")
        return ""
    
    try:
        recognized_text = "".join([word_info[1][0] for line in result for word_info in line if word_info and word_info[1]])
    except Exception as e:
        print("OCR 결과 처리 중 오류 발생:", e)
        recognized_text = ""
    return recognized_text