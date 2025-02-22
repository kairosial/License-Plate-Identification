import os
import re
import csv
from PIL import Image, ImageDraw, ImageFont

def annotate_from_detections(
    images_folder: str,
    recognized_text: str,
    detections_file: str = os.path.join('output', 'detection', 'detections.txt'),
    output_folder: str = os.path.join('output', 'OCR'),
    font_size: int = 32  # 원하는 폰트 크기 기본값(예: 32)
):
    """
    detections.txt 파일에 기록된 좌표 정보를 이용하여 원본 이미지에 바운딩 박스를 그리고,
    각 박스 위에 OCR 인식 결과(번호판 텍스트)를 표시한 주석 이미지를 생성한 후, output_folder에 저장합니다.

    :param images_folder: 원본 이미지들이 위치한 폴더 (예: "data/crop")
    :param recognized_text: OCR 인식 결과 문자열
    :param detections_file: 예: "data/crop/detections.txt"
    :param output_folder: 주석 이미지가 저장될 폴더 (예: "data/output")
    :param font_size: 그릴 텍스트의 폰트 크기 (기본값=32)
    :return: 주석 이미지 파일 경로 (성공 시), 없으면 None
    """
    os.makedirs(output_folder, exist_ok=True)

    def draw_korean_text(draw_obj, position, text, fill="yellow",
                         stroke_width=0, stroke_fill=None, font=None):
        """한글 텍스트를 이미지에 그립니다."""
        if font is None:
            font = ImageFont.load_default()
        x, y = position

        # 텍스트 외곽선 그리기 (선택)
        if stroke_width > 0 and stroke_fill:
            offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for ox, oy in offsets:
                draw_obj.text((x + ox, y + oy), text, font=font, fill=stroke_fill)
        draw_obj.text((x, y), text, font=font, fill=fill)

    def clean_filename(raw_filename: str) -> str:
        return os.path.basename(raw_filename.strip())

    # --- (1) detections.txt 파싱 ---
    detections_by_file = {}
    with open(detections_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            raw_filename = row["Filename"]
            filename = clean_filename(raw_filename)
            tag = row["Tag"]
            probability = float(row["Probability"])
            left = int(row["Left"])
            top = int(row["Top"])
            width = int(row["Width"])
            height = int(row["Height"])
            if filename not in detections_by_file:
                detections_by_file[filename] = []
            detections_by_file[filename].append({
                "tag": tag,
                "probability": probability,
                "left": left,
                "top": top,
                "width": width,
                "height": height
            })

    if not detections_by_file:
        print("detections.txt에 기록된 정보가 없습니다.")
        return None

    # --- (2) 첫 번째 이미지 사용 ---
    filename = list(detections_by_file.keys())[0]
    image_path = os.path.join(filename)  # images_folder가 필요하다면 os.path.join(images_folder, filename)으로 수정
    if not os.path.exists(image_path):
        print(f"이미지 파일 {image_path}이 존재하지 않습니다. 건너뜁니다.")
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # --- (3) 폰트 크기 적용 ---
    # 원하는 폰트(맑은 고딕 등)를 설정하거나 기본 폰트를 사용
    # Windows 맑은 고딕 예시
    try:
        if os.name == 'nt':
            # Windows
            font_path = "malgun.ttf"  # C:\Windows\Fonts\malgun.ttf가 있다면 절대경로로 지정 가능
        else:
            # Linux/Mac
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
        print("지정 폰트를 로딩할 수 없어 기본 폰트를 사용합니다.")

    # --- (4) 각 영역에 바운딩 박스 및 텍스트 ---
    for idx, det in enumerate(detections_by_file[filename]):
        left = det["left"]
        top = det["top"]
        w = det["width"]
        h = det["height"]
        box = (left, top, left + w, top + h)

        # 박스 그리기
        draw.rectangle(box, outline="red", width=4)
        # 텍스트 위치
        text_y = top - 40 if top - 40 > 0 else top

        # 텍스트 출력
        draw_korean_text(
            draw_obj=draw,
            position=(left, text_y),
            text=recognized_text,
            fill="yellow",
            stroke_width=10,
            stroke_fill="black",
            font=font
        )

    # --- (5) 결과 저장 ---
    annotated_filename = f"{os.path.splitext(filename)[0]}_annotated.jpg"
    annotated_path = os.path.join(output_folder, annotated_filename)
    image.save(annotated_path)
    print(f"주석 이미지 저장 완료: {annotated_path}")
    return annotated_path
