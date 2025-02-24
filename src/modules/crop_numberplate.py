import os
import io
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

CONFIG_PATH = os.path.join('src', 'modules', 'config.json')

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def crop_numberplate(
    image,
    output_folder: str = os.path.join('output', 'detection'),
    config_path: str = CONFIG_PATH
):
    """
    Azure Custom Vision을 사용하여 자동차 번호판을 감지하고 크롭하는 함수.
    파일 경로 대신 업로드된 이미지(PIL 또는 numpy 배열)를 직접 받아 처리합니다.
    """
    config = load_config(config_path)
    azure_cv = config["azure-cv"]

    endpoint = azure_cv["endpoint"]
    prediction_key = azure_cv["prediction_key"]
    project_id = azure_cv["project_id"]
    model_name = azure_cv["published_model_name"]

    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint, credentials)

    os.makedirs(output_folder, exist_ok=True)

    # 1) numpy 배열인지 확인 후, PIL 이미지로 변환
    if isinstance(image, np.ndarray):
        # image.shape: (H, W, C) 형식이어야 함 (RGB)
        image = Image.fromarray(image)

    # 2) Azure Custom Vision API가 요구하는 바이트 스트림으로 변환
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")  # PIL 이미지를 JPEG 포맷으로 메모리에 저장
    image_bytes.seek(0)

    # 3) 예측 요청
    results = predictor.detect_image(project_id, model_name, image_bytes.getvalue())
    if not results.predictions:
        print("번호판을 감지하지 못했습니다.")
        return []

    cropped_results = []
    width_img, height_img = image.size

    # fig, ax = plt.subplots(1, figsize=(8, 6))  # 시각화용 (필요 시 활성화)
    # ax.imshow(image)

    base_name = "uploaded_image"
    ext = ".jpg"

    for idx, prediction in enumerate(results.predictions):
        if prediction.probability > 0.8 and prediction.tag_name != "bus":
            left = int(prediction.bounding_box.left * width_img)
            top = int(prediction.bounding_box.top * height_img)
            crop_width = int(prediction.bounding_box.width * width_img)
            crop_height = int(prediction.bounding_box.height * height_img)

            if (crop_width * crop_height) / (width_img * height_img) < 0.02:
                continue

            # rect = patches.Rectangle((left, top), crop_width, crop_height, linewidth=2, edgecolor='red', facecolor='none')
            # ax.add_patch(rect)

            cropped_img = image.crop((left, top, left + crop_width, top + crop_height))
            if cropped_img.mode in ['RGBA', 'P']:
                cropped_img = cropped_img.convert('RGB')

            # 여기서 실제 파일로 저장하지 않고, 메모리 내 처리만 하고 싶다면 주석 처리 가능
            cropped_img_path = os.path.join(output_folder, f"{base_name}_cropped_{idx}{ext}")
            cropped_img.save(cropped_img_path)

            cropped_results.append({
                "cropped_image": cropped_img,
                "box": (left, top, crop_width, crop_height)
            })

    return cropped_results