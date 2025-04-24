# 📸 블랙박스 이미지 번호판 인식 시스템

## 📌 Summary

-   🎯 Automatically detect and recognize license plates from user-submitted dashcam images.
-   🤔 **Why** this project exists
    -   🚧 Many traffic violation reports in Korea’s Safety Report Portal(안전신문고) are rejected due to unreadable license plates in dashcam footage.
    -   🌙 Dashcam images often suffer from low-light, low-resolution, or motion blur.
    -   📣 A robust recognition system is needed to help users submit valid reports more easily.
-   🚀 **How** we solved it
    -   🛠️ Unified pipeline that enhances low-light images, detects object(license plates), and extracts text using OCR.
    -   🖼️ Provide a final output image with the recognized plate number overlaid as a caption.

## 👀 How does it work?

<img src="./images/demo.gif" style="width: 100%; height: auto;" />

## ⚙️ Pipeline

<img src="./images/pipeline.png" style="width: 100%; height: auto;" />
<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white" />
  <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img alt="PaddleOCR" src="https://img.shields.io/badge/PaddleOCR-0062B0?style=for-the-badge&logo=paddlepaddle&logoColor=white" />
  <img alt="Roboflow" src="https://img.shields.io/badge/Roboflow-6706CE?style=for-the-badge&logo=roboflow&logoColor=white" />
  <img alt="Azure" src="https://img.shields.io/badge/Azure-0089D6?style=for-the-badge&logoColor=white" />
  <img alt="Gradio" src="https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white" />
</p>
