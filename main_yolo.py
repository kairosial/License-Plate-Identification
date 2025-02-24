import os
import argparse
from src.modules.LLIE.colie_re import colie_re
from src.modules.detection.detect_and_crop import detect_numberplate_bbox, crop_numberplate_from_bbox
from src.modules.ocr_numberplate import ocr_numberplate
from src.modules.annotate_numberplate import annotate_from_detections
from ultralytics import YOLO

def main(args):
    # 프로젝트 루트를 기준으로 경로 설정 (main.py가 프로젝트 루트에 위치)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 입력 이미지(또는 이미지들이 담긴 디렉토리) 경로는 argparse로 전달받음
    image_path = args.image_path

    # LLIE 보정 출력 폴더 (PROJECT_ROOT/output/LLIE)
    llie_output_dir = os.path.join(CURRENT_DIR, "output", "LLIE")
    
    # LLIE 보정 여부 (옵션 --skip_llie가 지정되면 보정을 건너뜁니다.)
    if args.skip_llie:
        corrected_image = image_path
        print("LLIE 보정을 건너뜁니다.")
    else:
        corrected_image = colie_re(image_path, output_dir=llie_output_dir)
        if corrected_image is None:
            print("LLIE 보정이 적용되지 않아 원본 이미지를 사용합니다.")
            corrected_image = image_path

    # detection 결과 이미지와 crop된 이미지가 저장될 폴더 설정 (PROJECT_ROOT/output/detection/...)
    detection_save_dir = os.path.join(CURRENT_DIR, "output", "detection", "images")
    crop_output_folder  = os.path.join(CURRENT_DIR, "output", "detection", "cropped_numberplates")
    
    # 모델 경로: PROJECT_ROOT/output/detection/model/train/weights/best.pt
    model_path = os.path.join(CURRENT_DIR, "output", "detection", "model", "train", "weights", "best.pt")
    model = YOLO(model_path)
    
    # corrected_image가 디렉토리인 경우 해당 디렉토리 내부의 이미지 파일들을 처리
    image_files = []
    if os.path.isdir(corrected_image):
        allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(corrected_image, f) for f in os.listdir(corrected_image)
                       if f.lower().endswith(allowed_exts)]
    else:
        image_files = [corrected_image]
    
    for img_file in image_files:
        print(f"\nProcessing {img_file} ...")
        # 번호판 detection 및 detection 결과 이미지 저장
        detection_result = detect_numberplate_bbox(img_file, model, resize=True, detection_save_dir=detection_save_dir)
        if detection_result is None:
            print(f"번호판 감지 실패: {img_file}")
            continue
        bboxes, scale, original_img = detection_result
        image_basename = os.path.splitext(os.path.basename(img_file))[0]
        
        # detection 결과를 바탕으로 원본 이미지에서 번호판 영역 crop
        cropped_files = crop_numberplate_from_bbox(original_img, bboxes, scale, crop_output_folder, image_basename)
        if not cropped_files:
            print(f"번호판 영역 크롭 실패: {img_file}")
            continue
        crop_output_image = cropped_files[0]  # 여러 결과 중 첫 번째 사용
        print(f"크롭된 이미지 저장 완료: {crop_output_image}")
        
        # OCR 수행
        recognized_text = ocr_numberplate(crop_output_image)
        print(f"인식된 번호판 텍스트: {recognized_text}")
        
        # 주석 처리: 원본 이미지와 OCR 결과를 이용해 주석 이미지를 생성
        detections_file = os.path.join(CURRENT_DIR, "output", "detection", "detections.txt")
        ocr_output_folder = os.path.join(CURRENT_DIR, "output", "OCR")
        # annotated_image = annotate_from_detections()
        # if annotated_image:
        #     print(f"주석 처리된 이미지 저장 완료: {annotated_image}")
        # else:
        #     print("주석 이미지 생성 실패.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR 파이프라인 실행")
    parser.add_argument("--image_path", type=str, required=True, help="입력 이미지 파일 또는 이미지들이 담긴 디렉토리 경로")
    parser.add_argument("--skip_llie", action="store_true", help="LLIE 보정을 건너뜁니다.")
    args = parser.parse_args()
    main(args)