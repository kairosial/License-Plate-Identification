import os
import numpy as np
import torch
from PIL import Image

from utils import *
from loss import *
from siren import *
from color import *

# 이미지 파일 경로 및 출력 파일 경로를 하드코딩합니다.
input_image = 'input/image.jpg'
output_image = 'output/image_.jpg'

# 고정 파라미터 설정
down_size = 256
epochs = 100
window = 1
alpha = 1
beta = 20
gamma = 9
delta = 5
L_val = 0.1

# 밝기 임계값
brightness_threshold = 0.45

if not os.path.exists(input_image):
    print('입력 이미지 파일이 존재하지 않습니다: {}'.format(input_image))
    exit()

# 이미지 로드 및 HSV 변환
img_rgb = get_image(input_image)
img_hsv = rgb2hsv_torch(img_rgb)
img_v = get_v_component(img_hsv)

# 평균 밝기 계산 (0~1 범위라고 가정)
avg_brightness = torch.mean(img_v).item()
print(f"평균 밝기: {avg_brightness:.2f}")

# 밝기가 임계값 이상이면 보정 없이 원본 저장
if avg_brightness >= brightness_threshold:
    print("이미지가 충분히 밝아 colie 보정을 건너뜁니다.")
    Image.fromarray(
        (torch.movedim(img_rgb, 1, -1)[0].detach().cpu().numpy() * 255).astype(np.uint8)
    ).save(output_image)
else:
    print("어두운 이미지로 판단되어 colie 보정을 진행합니다.")
    # 보정을 위한 전처리
    img_v_lr = interpolate_image(img_v, down_size, down_size)
    coords = get_coords(down_size, down_size)
    patches = get_patches(img_v_lr, window)

    # SIREN 모델 초기화 (GPU 사용 부분: GPU가 없다면 .cuda()를 제거하거나 .to(device)로 수정하세요)
    img_siren = INF(patch_dim=window**2, num_layers=4, hidden_dim=256, add_layer=2)
    img_siren.cuda()

    optimizer = torch.optim.Adam(img_siren.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
    l_exp = L_exp(16, L_val)
    l_TV = L_TV()

    # 보정 최적화 진행
    for epoch in range(epochs):
        img_siren.train()
        optimizer.zero_grad()

        illu_res_lr = img_siren(patches, coords)
        illu_res_lr = illu_res_lr.view(1, 1, down_size, down_size)
        illu_lr = illu_res_lr + img_v_lr

        img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)

        loss_spa = torch.mean(torch.abs((illu_lr - img_v_lr) ** 2))
        loss_tv  = l_TV(illu_lr)
        loss_exp = torch.mean(l_exp(illu_lr))
        loss_sparsity = torch.mean(img_v_fixed_lr)

        loss = loss_spa * alpha + loss_tv * beta + loss_exp * gamma + loss_sparsity * delta
        loss.backward()
        optimizer.step()

    # 후처리 및 결과 이미지 저장
    img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v)
    img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
    img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)

    Image.fromarray(
        (torch.movedim(img_rgb_fixed, 1, -1)[0].detach().cpu().numpy() * 255).astype(np.uint8)
    ).save(output_image)

print(" > reconstruction done")