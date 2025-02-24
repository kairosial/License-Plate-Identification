import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from .utils import *
from .loss import *
from .siren import *
from .color import *

def colie_re(input_image, output_dir: str = os.path.join('output', 'LLIE')):
    """
    어두운 이미지를 보정하는 함수.
    :param input_image: 입력 PIL 이미지 객체 (RGB)
    :param output_dir: 결과 이미지를 저장할 폴더 (원한다면 저장 후 PIL 이미지 반환)
    :return: 보정된 PIL 이미지 객체 (RGB)
    """
    # input_image가 PIL 이미지가 아닐 경우 오류 처리
    if not isinstance(input_image, Image.Image):
        print("입력 이미지는 PIL 이미지 객체여야 합니다.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # 고정 파라미터 설정
    down_size = 256
    epochs = 100
    window = 1
    alpha = 1
    beta = 20
    gamma = 9
    delta = 5
    L_val = 0.1
    brightness_threshold = 0.45

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # PIL 이미지를 torch tensor로 변환 (값 범위: [0,1], shape: [C, H, W])
    transform = transforms.ToTensor()
    img_rgb = transform(input_image).unsqueeze(0).to(device)  # shape: [1, 3, H, W]
    
    # HSV 변환
    img_hsv = rgb2hsv_torch(img_rgb)
    img_v = get_v_component(img_hsv)
    
    # 평균 밝기 계산
    avg_brightness = torch.mean(img_v).item()
    print(f"평균 밝기: {avg_brightness:.2f}")
    
    # 밝기가 충분하면 보정 없이 원본을 반환
    if avg_brightness >= brightness_threshold:
        print("이미지가 충분히 밝아 colie 보정을 건너뜁니다.")
        # tensor -> PIL 이미지 변환
        img_np = (
            torch.movedim(img_rgb.squeeze(0), 0, -1)
            .detach()
            .cpu()
            .numpy() * 255
        ).astype(np.uint8)
        return Image.fromarray(img_np)
    
    print("어두운 이미지로 판단되어 colie 보정을 진행합니다.")
    
    # 보정 전처리
    img_v_lr = interpolate_image(img_v, down_size, down_size)
    coords = get_coords(down_size, down_size)
    patches = get_patches(img_v_lr, window)
    
    # SIREN 모델 초기화
    img_siren = INF(patch_dim=window**2, num_layers=4, hidden_dim=256, add_layer=2).to(device)
    optimizer = torch.optim.Adam(img_siren.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
    l_exp = L_exp(16, L_val)
    l_TV = L_TV()
    
    # 학습 수행
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
    
    # 후처리: tensor -> numpy -> PIL
    img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v)
    img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
    img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)
    
    img_np = (
        torch.movedim(img_rgb_fixed.squeeze(0), 0, -1)
        .detach()
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)
    output_image = Image.fromarray(img_np)
    
    # 저장 (선택 사항)
    output_path = os.path.join(output_dir, "colie_corrected.jpg")
    output_image.save(output_path)
    print(f"보정 완료: {output_path}")
    
    # 결과 PIL 이미지를 그대로 반환
    return output_image