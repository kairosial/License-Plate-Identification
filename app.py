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

def process_image_web(pil_image):
    if pil_image is None:
        return None

    # 1) 원본 이미지 백업
    if isinstance(pil_image, np.ndarray):
        # 만약 OpenCV BGR 형식이라면, BGR->RGB 변환 필요
        # 여기서는 RGB numpy라고 가정
        pil_image = Image.fromarray(pil_image)

    # 2) 원본 이미지 = PIL
    original_image = pil_image.copy()

    # 3) colie 보정
    colie_corrected_image = colie_re(original_image)  # 반환이 numpy일 수도 있음
    if isinstance(colie_corrected_image, np.ndarray):
        colie_corrected_image = Image.fromarray(colie_corrected_image)
    if colie_corrected_image is None:
        print("colie 보정 실패. 원본 이미지를 사용합니다.")
        colie_corrected_image = original_image

    # 3) YOLO 모델 로드 (1회만 해도 되지만, 예시로 매번 로드)
    model_path = "output/detection/model/train/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"YOLO model not found at {model_path}")
        return None
    model = YOLO(model_path)

    # 4) 바운딩 박스 감지 (메모리 방식)
    bboxes = detect_numberplates_in_image(
        colie_corrected_image,
        model=model,
        resize=True  # 640x640
    )
    if len(bboxes) == 0:
        print("번호판 감지 실패")
        return None

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

        width = x2 - x1
        height= y2 - y1
        detections_info.append({
            "box": (x1, y1, width, height),
            "text": recognized_text
        })

    # 6) 주석 처리
    annotated_image = annotate_from_detections(original_image, detections_info)
    if annotated_image is None:
        print("주석 이미지 생성 실패")
        return None

    # 7) BGR->RGB 변환 or RGB->BGR?
    # Gradio가 RGB numpy를 받는다면, 그냥 np.array(annotated_image) 반환해도 됨
    #annotated_np = np.array(annotated_image)  # PIL => RGB numpy
    # 만약 "cv2.imshow 등에서 BGR 필요"하면 다음 줄
    #annotated_rgb = cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB)

    return annotated_image

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