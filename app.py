import os
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import tempfile

# 위에서 수정한 함수들을 import 합니다.
from src.modules.LLIE.colie_re import colie_re 
from src.modules.ocr_numberplate import ocr_numberplate
from src.modules.crop_numberplate import crop_numberplate
from src.modules.annotate_numberplate import annotate_from_detections

def process_image_web(pil_image):
    if pil_image is None:
        return None

    original_image = pil_image.copy()
    # pil_image는 이미 RGB 상태의 PIL 이미지입니다.
    # 1. colie 보정: 파일 경로 없이 직접 이미지 객체를 받아 처리합니다.
    colie_corrected_image = colie_re(pil_image)
    
    if isinstance(colie_corrected_image, np.ndarray):
      colie_corrected_image = Image.fromarray(colie_corrected_image)
    
    if colie_corrected_image is None:
        print("colie 보정 실패. 원본 이미지를 사용합니다.")
        colie_corrected_image = pil_image

    # 2. 번호판 크롭: 보정된 이미지를 사용하여 번호판 영역 검출 (PIL 이미지 객체 입력)
    crop_results = crop_numberplate(colie_corrected_image)
    if not crop_results:
        print("번호판을 감지하지 못했습니다.")
        return None

    # 3 각 번호판마다 OCR 수행
    # box + recognized_text 정보를 별도로 저장
    detections_info = []
    for i, result in enumerate(crop_results):
        cropped_image = result["cropped_image"]  # PIL 이미지
        text = ocr_numberplate(cropped_image)
        print(f"번호판 {i+1} 텍스트: {text}")

        # box 정보
        left, top, width, height = result["box"]
        detections_info.append({
            "box": (left, top, width, height),
            "text": text
        })

    # 4. 주석 처리: 보정된 원본 이미지, OCR 텍스트, 검출 결과(바운딩 박스 정보)를 사용하여 주석 이미지 생성
    annotated_image = annotate_from_detections(original_image, detections_info)
    if annotated_image is None:
        print("주석 이미지 생성에 실패하였습니다.")
        return None

    # BGR -> RGB 변환
    annotated_image_np = np.array(annotated_image)
    annotated_image_rgb = cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb

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
"""

with gr.Blocks(css=custom_css) as demo:
    # 개인정보 확인 페이지: 아직 확인 기록이 없으면 보임
    with gr.Column() as confirm_container:
         gr.Markdown("<div style='text-align: center;'><H1>License Plate Identification<H1></div>")
         gr.Markdown("### 개인정보 관련 내용")
         gr.Markdown('''  
개인정보 보호법 제2조 1호 가목, 나목  
1."개인정보"란 살아 있는 개인에 관한 정보로서 다음 각 목의 하나에 해당하는 정보를 말한다.  
가. 성명, 주민등록번호 및 영상 등을 통하여 개인을 알아볼 수 있는 정보  
나. 해당 정보만으로는 특정 개인을 알아볼 수 없더라도 다른 정보와 쉽게 결합하여 알아볼 수 있는 정보. (이경우 다른 정보의 입수 가능성 등 개인을 알아보는데 소요되는 시간, 비용, 기술 등을 합리적으로 고려)  

즉, 자동차등록번호는 자동차관리법에 따라 자동차에 부여된 일련번호로 일반적으로는 개인정보가 아니지만, 다른 정보와 쉽게 결합하여 개인을 알아볼 수 있는 특수한 상황에서는 개인정보에 해당할 수 없습니다.
         ''')
         confirm_button = gr.Button("확인")
         
    # 메인 인터페이스: 개인정보 확인 후 보임
    with gr.Column(visible=False) as main_container:
         gr.Markdown("<div style='text-align: center;'><H1>License Plate Identification<H1></div>")
         with gr.Row():
             input_image = gr.Image(type="numpy", label="번호판 이미지 업로드", sources="upload")
             output_image = gr.Image(type="numpy", label="주석 처리된 이미지")
         # 이미지를 업로드하면 바로 처리
         input_image.change(fn=process_image_web, inputs=input_image, outputs=output_image)
         with gr.Row(elem_classes="center-container"):
             clear_button = gr.Button("Clear", elem_classes="clear-button")
         clear_button.click(fn=clear_all, inputs=[], outputs=[input_image, output_image])
         
    # 개인정보 확인 버튼 클릭 시 처리하는 함수
    def confirm_action():
         # 개인정보 확인 컨테이너는 숨기고 메인 인터페이스 컨테이너를 보이도록 업데이트
         return gr.update(visible=False), gr.update(visible=True)
    
    confirm_button.click(fn=confirm_action, inputs=[], outputs=[confirm_container, main_container])

demo.launch(share=True, server_name="0.0.0.0", server_port=7956)