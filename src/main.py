import os
from modules.ocr_numberplate import ocr_numberplate  # 함수만 import

def main():
    # 사용자로부터 이미지 파일 경로 입력 받기
    image_path = input("번호판 이미지 파일 경로를 입력하세요: ")
    
    # 이미지 경로가 존재하는지 확인
    if not os.path.exists(image_path):
        print("지정한 경로에 파일이 존재하지 않습니다.")
        return
    
    # 번호판 인식
    recognized_text = ocr_numberplate(image_path)  # 함수로 호출
    
    # 결과 출력
    print(f"인식된 번호판 텍스트: {recognized_text}")

if __name__ == "__main__":
    main()
