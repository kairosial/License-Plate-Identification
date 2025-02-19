import cv2
import numpy as np
import matplotlib.pyplot as plt

# 클릭한 좌표를 저장할 리스트
points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        points.append((int(event.xdata), int(event.ydata)))
        print(f"좌표: ({int(event.xdata)}, {int(event.ydata)})")

        # 총 4번 클릭했으면 종료
        if len(points) == 4:
            plt.close()

# 이미지 로드
image_path = 'input.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 표시
fig, ax = plt.subplots()
ax.imshow(image_rgb)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

print("클릭한 좌표:", points)
