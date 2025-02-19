import cv2
import numpy as np

def transform_image(image_path, points):
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 입력 이미지의 네 개의 좌표 (왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래)
    pts_src = np.array(points, dtype=np.float32)
    
    # 변환 후의 이미지 크기 설정
    width = max(int(np.linalg.norm(pts_src[0] - pts_src[1])), int(np.linalg.norm(pts_src[2] - pts_src[3])))
    height = max(int(np.linalg.norm(pts_src[0] - pts_src[3])), int(np.linalg.norm(pts_src[1] - pts_src[2])))
    
    # 목적 좌표 설정
    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    # 이미지 변환
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return transformed_image

# 예제 사용법
image_path = 'input.png'
points = [#(82, 269), (256, 92), (316, 228), (139, 432)
    [82, 269],  # 왼쪽 위
    [256, 92],  # 오른쪽 위
    [316, 228],  # 오른쪽 아래
    [139, 432]   # 왼쪽 아래
]
transformed_image = transform_image(image_path, points)

# 결과 이미지 저장
cv2.imwrite('output_image.png', transformed_image)
