import os
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO

# 위에서 수정한 함수들을 import 합니다.
from src.modules.LLIE.colie_re import colie_re 
from src.modules.ocr_numberplate import ocr_numberplate
#from src.modules.crop_numberplate import crop_numberplate
from src.modules.annotate_numberplate import annotate_from_detections
from src.modules.detection.detect_and_crop import detect_numberplates_in_image

def process_image_web(pil_image, result_text):
    # 프로젝트 루트를 기준으로 경로 설정 (main.py가 프로젝트 루트에 위치)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    result_text = ""
    # 1) 업로드한 이미지가 올바른 형식인지 확인 (이미지 파일이 아닌 경우)
    if pil_image is None:
        return None, "이미지 파일 업로드해주세요"
    
    if not isinstance(pil_image, Image.Image):
        try:
            pil_image = Image.fromarray(pil_image)
        except Exception:
            return None, "이미지 파일 업로드해주세요"
    
    if isinstance(pil_image, np.ndarray):
        # 만약 OpenCV BGR 형식이라면, BGR->RGB 변환 필요
        # 여기서는 RGB numpy라고 가정
        pil_image = Image.fromarray(pil_image)

    # 2) 원본 이미지 = PIL 백업
    original_image = pil_image.copy()

    # 3) colie 보정
    colie_corrected_image = colie_re(original_image)  # 반환이 numpy일 수도 있음
    if isinstance(colie_corrected_image, np.ndarray):
        colie_corrected_image = Image.fromarray(colie_corrected_image)
    
    if colie_corrected_image is None:
        print("밝기가 밝아 LLIE 보정 없이 원본 이미지를 사용합니다.")
        colie_corrected_image = original_image

    # 3) YOLO 모델 로드 (1회만 해도 되지만, 예시로 매번 로드)
    model_path = os.path.join(CURRENT_DIR, "output", "detection", "model", "train", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"{model_path} 위치에 모델를 발견하지 못했습니다.")
        return None, "모델 파일을 찾을 수 없습니다."
    model = YOLO(model_path)

    # 4) 바운딩 박스 감지 (메모리 방식)
    bboxes = detect_numberplates_in_image(
        colie_corrected_image,
        model=model,
        resize=True  # 640x640
    )
    if len(bboxes) == 0:
        print("차량을 감지하지 못했습니다.")
        return None, "차량을 감지하지 못했습니다"

    # 5) 바운딩 박스별 OCR
    detections_info = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box
        # int 변환
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # 크롭 (인메모리)
        cropped_img = original_image.crop((x1, y1, x2, y2))
        recognized_text = ocr_numberplate(cropped_img)

        print(f"[{i+1}] OCR result: {recognized_text}")

        if not recognized_text or recognized_text.strip() == "":
            return None, "차량 번호판 인식하는데에 있어 실패하였습니다."
        
        width = x2 - x1
        height= y2 - y1
        detections_info.append({
            "box": (x1, y1, width, height),
            "text": recognized_text
        })

    # 6) 주석 처리
    annotated_result = annotate_from_detections(original_image, detections_info)
    if annotated_result is None:
        print("주석 이미지 생성을 실패했습니다.")
        return None, "주석 이미지 생성에 실패하였습니다."

    # 만약 annotate_from_detections가 tuple을 반환한다면 첫 번째 요소를 실제 이미지로 사용
    if isinstance(annotated_result, tuple):
        annotated_image = annotated_result[0]
    else:
        annotated_image = annotated_result

    result_text = "차량 번호판 인식을 성공하였습니다."
    return annotated_image, result_text

def clear_all():
    return None, None

# Clear 버튼과 관련된 CSS (버튼을 양옆으로 넓게, 가운데 정렬)
custom_css = """
.clear-button {
    width: 100%;
    font-size: 20px;
    padding: 20px;
    margin-top: 20px;
}
.center-container {
    display: flex;
    justify-content: center;
}

.centered-textbox textarea {
    text-align: center;
    font-size: 20px;
    height: 40px;      
    padding-top: 10px; 
    padding-bottom: 2px; 
}
"""
with gr.Blocks(css=custom_css) as demo:       
    # 메인 인터페이스
    with gr.Column() as main_container:
         gr.Markdown("<div style='text-align: center;'><H1>License Plate Identification<H1></div>")
         
         with gr.Row():
             input_image = gr.Image(type="numpy", label="번호판 이미지 업로드", sources="upload")
             output_image = gr.Image(type="numpy", label="주석 처리된 이미지")

         with gr.Row(elem_classes="center-container"):
            result_text = gr.Textbox(label="결과 메시지", show_label=True, lines=2, max_lines=2, elem_classes="centered-textbox")
         # 이미지 업로드 시 process_image_web 함수 실행 (이미지와 결과 메시지를 함께 반환
         input_image.change(fn=process_image_web, inputs=input_image, outputs=[output_image, result_text])
         with gr.Row(elem_classes="center-container"):
             clear_button = gr.Button("Clear", elem_classes="clear-button")
         clear_button.click(fn=clear_all, inputs=[], outputs=[input_image, output_image])
    # 개인정보 관련 사항
    with gr.Column() as confirm_container:
         gr.Markdown("<div style='text-align: center;'>본 서비스를 이용하는 순간, 귀하는 차량번호판 정보에 한정된 개인정보의 수집 및 활용에 대해 사전 동의한 것으로 간주됩니다.</div>")       

demo.launch(share=True, server_name="0.0.0.0", server_port=7956)