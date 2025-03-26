# Document Classifier (Estácio/UNIASSELVI) and Answer Sheet Identifier ✅

This application classifies uploaded documents (PDFs, DOC/DOCX, or images) as:
- **Estácio Exam**
- **Uniasselvi Exam**
- **Other**

It also detects checkmark boxes using a YOLO model, displaying them with green bounding boxes and showing statistics like count and average confidence.

## Features

- **File Conversion**: DOC/DOCX to PDF, PDF to images.
- **Document Classification**: Uses a LeViT model (ONNX format) to identify the type of document.
- **Checkmark Detection**: YOLO model identifies checkmarks with confidence thresholding and non-maximum suppression.
- **Streamlit Interface**: Clean UI with sidebar model selection, image previews, and detection stats.

## Installation

### System Requirements

- Python 3.10+
- Poppler for `pdf2image`
- LibreOffice (for DOC/DOCX conversion)

Install dependencies (after cloning the repository):

```bash
sudo apt update
sudo apt install poppler-utils libreoffice -y
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py --server.fileWatcherType none
```

Access the app at `http://localhost:8501`.

## Running on AWS EC2

1. Launch an EC2 instance with Ubuntu.
2. Install dependencies as described above.
3. Clone this repository and navigate into it.
4. Start the Streamlit app with:

```bash
streamlit run app.py --server.headless true --server.port 8501 --server.enableCORS false
```

5. Open your browser to `http://<your-ec2-public-ip>:8501`.

## Folder Structure

```
.
├── app.py
├── requirements.txt
└── models/
    ├── levit-384-exams-classification/
    │   ├── onnx/levit384_estacio-multiclass.onnx
    │   ├── preprocessor_config.json
    │   └── metrics/metrics.png
    └── yolo_detection_checkmark-final/
        └── best.pt
```

## License

Apache License

---

Developed by **Lucas Meireles**
