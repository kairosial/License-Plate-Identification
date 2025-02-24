import os
from src.modules.detection.detect_and_crop import detect_numberplate_bbox, crop_numberplate_from_bbox
from ultralytics import YOLO

def main():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    detection_save_dir = os.path.join(CURRENT_DIR, "output", "detection", "images")
    crop_output_folder = os.path.join(CURRENT_DIR, "output", "detection", "cropped_numberplates")
    model_path = os.path.join(CURRENT_DIR, "output", "detection", "model", "train", "weights", "best.pt")

    model = YOLO(model_path)
    
    # 처리할 이미지(또는 이미지들이 담긴 디렉토리) 경로
    image_path = '/home/azureuser/cloudfiles/code/Users/6b011/LPR/Azure-Data/blackbox-test'
    
    # image_path가 디렉토리인 경우, 해당 폴더 내의 이미지 파일들을 처리
    if os.path.isdir(image_path):
        allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path)
                       if f.lower().endswith(allowed_exts)]
    else:
        image_files = [image_path]
    
    for img_file in image_files:
        print(f"\nProcessing {img_file} ...")
        detection_result = detect_numberplate_bbox(img_file, model, resize=True, detection_save_dir=detection_save_dir)
        if detection_result is None:
            print(f"Detection failed for {img_file}.")
            continue
        
        bboxes, scale, original_img = detection_result
        image_basename = os.path.splitext(os.path.basename(img_file))[0]
        cropped_files = crop_numberplate_from_bbox(original_img, bboxes, scale, crop_output_folder, image_basename)
        print("Cropped files:", cropped_files)

if __name__ == "__main__":
    main()
