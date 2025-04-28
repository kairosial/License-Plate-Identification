# 📸 블랙박스 이미지 번호판 인식 시스템

## 📌 Summary

-   🎯 Automatically **detect and recognize license plates** from user-submitted **dashcam images**.
-   🤔 **Why** this project exists
    -   🚧 Many traffic violation reports in Korea’s Safety Report Portal(안전신문고) are rejected due to unreadable license plates in dashcam footage.
    -   🌙 Dashcam images often suffer from low-light, low-resolution, or motion blur.
    -   📣 A robust recognition system is needed to help users submit valid reports more easily.
-   🚀 **How** we solved it
    -   🛠️ Unified pipeline that combines **low-light image enhancement**, **object detection**, and **OCR**.
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

## 🧑🏻‍💻 Developers

<div style="overflow-x: auto;">
  <table align="center">
    <tr>
      <td align="center"><img src="https://github.com/kairosial.png" style="width: 90%; height: auto;" /></td>
      <td align="center"><img src="https://github.com/LioKim02.png" style="width: 90%; height: auto;" /></td>
      <td align="center"><img src="https://github.com/jooeun921.png" style="width: 90%; height: auto;" /></td>
      <td align="center"><img src="https://github.com/alice-min10.png" style="width: 90%; height: auto;" /></td>
      <td align="center"><img src="https://github.com/Ju-hong" style="width: 90%; height: auto;" /></td>
      <td align="center"><img src="https://github.com/jiiiiiiiy" style="width: 90%; height: auto;" /></td>
      <td align="center"><img src="https://github.com/hanx0419" style="width: 90%; height: auto;" /></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/kairosial">김승연</a></td>
      <td align="center"><a href="https://github.com/LioKim02">김일생</a></td>
      <td align="center"><a href="https://github.com/jooeun921">박주은</a></td>
      <td align="center"><a href="https://github.com/alice-min10">이수민</a></td>
      <td align="center"><a href="https://github.com/Ju-hong">이주홍</a></td>
      <td align="center"><a href="https://github.com/jiiiiiiiy">이지혜</a></td>
      <td align="center"><a href="https://github.com/hanx0419">허한성</a></td>
    </tr>
    <tr>
      <td align="center" style="min-width: 220px;">
        <div>Project planning</div>
        <div>Architecture design</div>
        <div>Code integration</div>
      </td>
      <td align="center" style="min-width: 220px;">
        <div>Server integration</div>
        <div>Super-Resolution</div>
        <div>Image transformation</div>
      </td>
      <td align="center" style="min-width: 220px;">
        <div>OCR</div>
        <div>Code integration</div>
        <div>Presentation design</div>
      </td>
      <td align="center" style="min-width: 220px;">
        <div>Object detection</div>
        <div>Data preprocessing</div>
        <div>Demo video</div>
      </td>
      <td align="center" style="min-width: 220px;">
        <div>OCR</div>
        <div>Azure OCR</div>
        <div>Presentation design</div>
      </td>
      <td align="center" style="min-width: 220px;">
        <div>Object detection</div>
        <div>Data preprocessing</div>
      </td>
      <td align="center" style="min-width: 220px;">
        <div>LLIE</div>
        <div>Gradio</div>
        <div>UI/UX</div>
      </td>
    </tr>
  </table>
</div>
