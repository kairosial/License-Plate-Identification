import os
from modules.ocr_numberplate import ocr_numberplate  # 번호판 인식 함수
from modules.colie.colie_re import colie_re          # colie 보정 함수
from modules.crop_numberplate import crop_numberplate  # 번호판 크롭 함수
from modules.annotate_numberplate import annotate_from_detections # 번호판 주석처리리

def main():
    # 사용자로부터 이미지 파일 경로 입력 받기
    user_image = input("번호판 이미지 파일 경로를 입력하세요: ")
    
    # 이미지 파일 존재 여부 확인
    if not os.path.exists(user_image):
        print("지정한 경로에 파일이 존재하지 않습니다.")
        return
    
    # colie 보정 적용 및 결과 이미지 경로 반환
    colie_output = colie_re(user_image)
    
    # 만약 colie_re가 None을 반환하면(즉, 보정을 건너뛰면)
    # 원본 이미지를 사용하도록 설정합니다.
    if colie_output is None:
        print("colie 보정이 적용되지 않아 원본 이미지를 사용합니다.")
        colie_output = user_image
    
    # 번호판 크롭 진행
    crop_output_image = crop_numberplate(colie_output)
    if not crop_output_image:
        print("번호판을 감지하지 못했습니다.")
        return
    else:
        # crop_numberplate가 리스트를 반환하는 경우, 첫 번째 결과를 사용합니다.
        crop_output_image = crop_output_image[0]
        print(f"크롭된 이미지 저장 완료: {crop_output_image}")
    
    # 번호판 인식 수행
    recognized_text = ocr_numberplate(crop_output_image)
    print(f"인식된 번호판 텍스트: {recognized_text}")

    #번호판 주석처리
    annotated_image = annotate_from_detections(user_image, recognized_text)
    if annotated_image:
        print("주석 처리된 이미지 저장 완료:", annotated_image)
    else:
        print("주석 이미지 생성에 실패하였습니다.")

if __name__ == "__main__":
    main()
