import os
from modules.ocr_numberplate import ocr_numberplate  # 함수만 import
from modules.colie import colie
from modules.crop_numberplate import crop_numberplate


def main():
    # 사용자로부터 이미지 파일 경로 입력 받기
    user_image = input("번호판 이미지 파일 경로를 입력하세요: ")
    
    # 이미지 경로가 존재하는지 확인
    if not os.path.exists(user_image):
        print("지정한 경로에 파일이 존재하지 않습니다.")
        return
    
    # 밝기 보정 적용 및 결과 이미지 경로 반환
    colie_output = colie(user_image)

    # 이미지 크롭하기
    crop_output_image = crop_numberplate(colie_output)

    if crop_output_image:
        print(f"크롭된 이미지 저장 완료: {crop_output_image}")
    else:
        print("번호판을 감지하지 못했습니다.")

    # 번호판 인식
    recognized_text = ocr_numberplate(crop_output_image)  # 보정된 이미지 사용
    
    # 결과 출력
    print(f"인식된 번호판 텍스트: {recognized_text}")

if __name__ == "__main__":
    main()
