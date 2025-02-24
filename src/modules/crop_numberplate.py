import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# 설정 파일 경로 (필요 시 수정)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config.json')

def load_config(config_path):
    """JSON 파일에서 Azure 설정값을 로드하는 함수"""
    with open(config_path, "r") as f:
        return json.load(f)

def crop_numberplate(image_path: str, output_folder: str = os.path.join('output', 'detection'), config_path: str = CONFIG_PATH):
    """
    Azure Custom Vision을 사용하여 자동차 번호판을 감지하고 크롭하는 함수.

    :param image_path: 입력 이미지 경로
    :param output_folder: 크롭된 이미지 저장 폴더
    :param config_path: Azure 설정 정보가 담긴 JSON 파일 경로
    :return: 크롭된 이미지 파일 경로 리스트
    """

    # 설정 불러오기
    config = load_config(CONFIG_PATH)
    azure_cv = config["azure-cv"]

    endpoint = azure_cv["endpoint"]
    prediction_key = azure_cv["prediction_key"]
    project_id = azure_cv["project_id"]
    model_name = azure_cv["published_model_name"]

    # Azure Custom Vision 클라이언트 생성
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint, credentials)

    # 결과 저장 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 바운딩 박스 좌표 저장 파일
    output_txt = os.path.join(output_folder, "detections.txt")

    cropped_image_paths = []
    with open(output_txt, "w") as f:
        f.write("Filename,Tag,Probability,Left,Top,Width,Height\n")  # 헤더 추가

        # 이미지 로드 및 예측
        with open(image_path, "rb") as image_data:
            results = predictor.detect_image(project_id, model_name, image_data)

        # 만약 감지 결과가 없으면 바로 반환
        if not results.predictions:
            print("번호판을 감지하지 못했습니다.")
            return []

        # 원본 이미지 로드
        image = Image.open(image_path)
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(image)

        base_name, ext = os.path.splitext(os.path.basename(image_path))

        # 바운딩 박스 처리
        for idx, prediction in enumerate(results.predictions):
            if prediction.probability > 0.8 and prediction.tag_name != "bus":
                # 바운딩 박스 좌표 변환 (비율 → 픽셀)
                left = int(prediction.bounding_box.left * image.width)
                top = int(prediction.bounding_box.top * image.height)
                width = int(prediction.bounding_box.width * image.width)
                height = int(prediction.bounding_box.height * image.height)

                # 크롭된 이미지 크기가 전체의 2% 이상인지 확인
                if (width * height) / (image.width * image.height) < 0.02:
                    continue

                # 바운딩 박스 시각화
                rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(left, top - 5, f"{prediction.tag_name} ({prediction.probability:.2f})",
                        bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

                # 크롭된 이미지 저장
                cropped_img = image.crop((left, top, left + width, top + height))
                if cropped_img.mode in ['RGBA', 'P']:
                    cropped_img = cropped_img.convert('RGB')

                cropped_img_path = os.path.join(output_folder, f"{base_name}_cropped_{idx}{ext}")
                cropped_img.save(cropped_img_path)
                cropped_image_paths.append(cropped_img_path)

                # 바운딩 박스 정보 저장
                f.write(f"{os.path.basename(image_path)},{prediction.tag_name},{prediction.probability:.2f},{left},{top},{width},{height}\n")

    return cropped_image_paths