import streamlit as st
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import tempfile
import subprocess
import shlex
import numpy as np
import onnxruntime as ort
import torch
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from ultralytics import YOLO
from transformers import ViTImageProcessor

# Set page configuration
st.set_page_config(
    page_title="Document Classifier (Estácio/UNIASSELVI) and Answer Sheet Identifier ✅",
    page_icon="📄",
    layout="wide"
)

st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 36px;
        color: #f48c5d;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #333333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .main-text {
        font-size: 16px;
        color: #333333;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #f48c5d;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Functions for file conversion

def docx_to_pdf(docx_path, pdf_path):
    try:
        output_dir = os.path.dirname(pdf_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        command = f'libreoffice --headless --convert-to pdf "{docx_path}" --outdir "{output_dir}"'
        subprocess.run(shlex.split(command), check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting .docx to PDF: {e}")
        raise

def doc_to_docx(input_path, output_path):
    try:
        subprocess.run(['libreoffice', '--headless', '--convert-to', 'docx', input_path, '--outdir', os.path.dirname(output_path)], check=True)
    except Exception as e:
        st.error(f"Error converting .doc to .docx: {e}")
        raise

def validate_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f, strict=True)
            num_pages = len(reader.pages)
            if num_pages == 0:
                raise ValueError("The PDF is empty.")
            return num_pages
    except Exception as e:
        raise ValueError(f"Error validating PDF: {e}")

def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, dpi=300)


# Function to filter bounding boxes
# Remove the box with the lower score if:
# - One box is contained within another, or
# - Two boxes "touch" (top and bottom borders are very close)

def filter_bboxes(bboxes, tolerance=5):
    n = len(bboxes)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            box_i = bboxes[i]
            box_j = bboxes[j]
            # Check if box i is contained in box j
            if (box_j['x_min'] <= box_i['x_min'] and box_j['y_min'] <= box_i['y_min'] and
                box_j['x_max'] >= box_i['x_max'] and box_j['y_max'] >= box_i['y_max']):
                if box_i['score'] < box_j['score']:
                    keep[i] = False
                    break  # No need to compare box i with others
                else:
                    keep[j] = False
                    continue
            # Check if box j is contained in box i
            if (box_i['x_min'] <= box_j['x_min'] and box_i['y_min'] <= box_j['y_min'] and
                box_i['x_max'] >= box_j['x_max'] and box_i['y_max'] >= box_j['y_max']):
                if box_j['score'] < box_i['score']:
                    keep[j] = False
                    continue
                else:
                    keep[i] = False
                    break
            # Check if the boxes "touch" (top and bottom borders are very close)
            if (abs(box_i['y_min'] - box_j['y_min']) <= tolerance and 
                abs(box_i['y_max'] - box_j['y_max']) <= tolerance):
                if box_i['score'] < box_j['score']:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
                    continue
    filtered = [bboxes[i] for i in range(n) if keep[i]]
    return filtered


# Function to compute Intersection over Union (IoU) between two boxes

def iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) between two boxes.
    boxA and boxB are dictionaries with keys: x_min, y_min, x_max, y_max.
    """
    xA = max(boxA['x_min'], boxB['x_min'])
    yA = max(boxA['y_min'], boxB['y_min'])
    xB = min(boxA['x_max'], boxB['x_max'])
    yB = min(boxA['y_max'], boxB['y_max'])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    areaA = (boxA['x_max'] - boxA['x_min']) * (boxA['y_max'] - boxA['y_min'])
    areaB = (boxB['x_max'] - boxB['x_min']) * (boxB['y_max'] - boxB['y_min'])
    
    union = areaA + areaB - interArea
    if union == 0:
        return 0
    return interArea / union


# Function to check if two boxes touch or overlap based on a pixel threshold

def boxes_touch_or_overlap(boxA, boxB, pixel_threshold=5):
    """
    Checks if two boxes "touch" vertically
    (top and bottom borders within pixel_threshold) OR
    are horizontally very close.
    Adjust the criteria as needed.
    """
    # Check if horizontally close
    horizontal_close = (
        abs(boxA['x_min'] - boxB['x_min']) <= pixel_threshold and
        abs(boxA['x_max'] - boxB['x_max']) <= pixel_threshold
    )
    # Check if vertically close
    vertical_close = (
        abs(boxA['y_min'] - boxB['y_min']) <= pixel_threshold and
        abs(boxA['y_max'] - boxB['y_max']) <= pixel_threshold
    )
    return horizontal_close or vertical_close


# Function to apply strict Non-Maximum Suppression (NMS) on boxes,
# removing those that have IoU >= iou_threshold OR that "touch" (as defined by pixel_threshold).
# Returns the list of filtered boxes.

def nms_strict(bboxes, iou_threshold=0.1, pixel_threshold=5):
    # Sort boxes by descending score
    bboxes = sorted(bboxes, key=lambda x: x['score'], reverse=True)
    keep = []
    discarded = set()

    for i in range(len(bboxes)):
        if i in discarded:
            continue
        current_box = bboxes[i]
        keep.append(current_box)
        for j in range(i + 1, len(bboxes)):
            if j in discarded:
                continue
            other_box = bboxes[j]
            # Calculate IoU
            iou_value = iou(current_box, other_box)
            # Check if boxes are "close enough"
            close_enough = boxes_touch_or_overlap(current_box, other_box, pixel_threshold)
            # If IoU is high or they touch, discard the box with lower score (other_box, since list is sorted)
            if iou_value >= iou_threshold or close_enough:
                discarded.add(j)
    return keep


# Function to draw YOLO bounding boxes (only for checkmarks in green)

def draw_yolo_bboxes(image, yolo, conf_threshold=0.5):
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    results = yolo.predict(source=np.array(image), conf=conf_threshold)
    candidate_boxes = []

    if not results or not hasattr(results[0], 'boxes'):
        return image_draw, []
    
    # Collect candidate boxes for "checkmark" detections
    for box in results[0].boxes:
        coords = box.xyxy.tolist()[0]
        x_min, y_min, x_max, y_max = map(int, coords)
        class_id = int(box.cls[0])
        class_name = yolo.names[class_id]
        if class_name.lower() == "checkmark":
            conf_value = float(box.conf[0]) if hasattr(box.conf, '__getitem__') else float(box.conf)
            candidate_boxes.append({
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'score': conf_value
            })
    
    # Apply strict NMS filtering
    filtered_boxes = nms_strict(candidate_boxes, iou_threshold=0.1, pixel_threshold=5)
    
    # Draw only the filtered boxes
    checkmark_confidences = []
    for fb in filtered_boxes:
        checkmark_confidences.append(fb['score'])
        draw.rectangle([fb['x_min'], fb['y_min'], fb['x_max'], fb['y_max']], outline="green", width=3)
        draw.text((fb['x_min'], fb['y_min']), f"checkmark ({fb['score']:.2f})", fill="green")

    return image_draw, checkmark_confidences


# Functions for classification with LeViT

def process_and_classify(images, processor, onnx_model_path, selected_model_name):
    ort_session = ort.InferenceSession(onnx_model_path)
    probabilities = []
    for image in images:
        inputs = processor(images=image, return_tensors="np")
        input_data = inputs["pixel_values"]
        ort_inputs = {"input": input_data}
        ort_outputs = ort_session.run(None, ort_inputs)
        logits = ort_outputs[0]
        if (selected_model_name == "LeViT Multiclass Model Fine-Tuned (Baseline) v1") or (selected_model_name == "LeViT 384 Multiclass Model Fine-Tuned"):
            probs = torch.softmax(torch.tensor(logits), axis=1).numpy()
        else:
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
        probabilities.append(probs)
    return np.mean(probabilities, axis=0)

def classify_with_threshold(probabilities, min_probability=0.70):
    max_score = np.max(probabilities)
    max_index = int(np.argmax(probabilities))
    # If confidence is almost 1.0 and the index is not "Other" (assuming "Other" corresponds to index 1)
    if np.isclose(max_score, 1.0, atol=1e-6) and max_index != 1:
         # Emit warning and classify as "Other"
         return 1
    # For indices 0 (Estácio Exam) or 2 (Uniasselvi Exam), check for minimum confidence
    if max_index in [0, 2]:
         if max_score >= min_probability:
              return max_index
         else:
              st.warning("Low confidence.")
              return 1
    return max_index


# LeViT Model configuration in the sidebar

st.sidebar.header("Select LeViT Model")

model_descriptions = {
    "LeViT 384 Multiclass Model Fine-Tuned": (
        "The **LeViT 384 Multiclass Model Fine-Tuned** was trained with a dataset containing exams from Estácio, UNIASSELVI, and others. "
        "It was trained for 5 epochs with a weight decay of 0.05 and achieved good metrics. "
        "This version uses more convolutional layers and attention mechanisms."
    )
}

model_paths = {
    "LeViT 384 Multiclass Model Fine-Tuned": "models/levit-384-exams-classification/onnx/levit384_estacio-multiclass.onnx"
}

selected_model_name = st.sidebar.selectbox(
    "Select a Model",
    options=list(model_paths.keys()),
    index=0
)
selected_model_path = model_paths[selected_model_name]

if selected_model_name == "LeViT 384 Multiclass Model Fine-Tuned":
    processor = ViTImageProcessor.from_pretrained("models/levit-384-exams-classification/preprocessor_config.json")
if st.sidebar.button("Load Model"):
    st.sidebar.success(f"Model '{selected_model_name}' loaded successfully!")

with st.expander("📊 Selected Model Metrics"):
    st.markdown(f"**Name:** {selected_model_name}")
    st.markdown(model_descriptions[selected_model_name])
    if selected_model_name == "LeViT 384 Multiclass Model Fine-Tuned":
        st.image("models/levit-384-exams-classification/metrics/metrics.png")


# Load YOLO model

yolo = YOLO('models/yolo_detection_checkmark-final/best.pt')


# Title and file uploader

st.markdown('<h1 class="main-header">Document Classifier (Estácio/UNIASSELVI) and Answer Sheet Identifier ✅</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF, DOC/DOCX or Image:", type=["pdf", "doc", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        images = []
        
        if file_type == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name
            validate_pdf(temp_pdf_path)
            images = convert_pdf_to_images(temp_pdf_path)
        
        elif file_type in ["doc", "docx"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_doc:
                temp_doc.write(uploaded_file.read())
                temp_doc_path = temp_doc.name
            if file_type == "doc":
                new_docx_path = temp_doc_path.replace(".doc", ".docx")
                doc_to_docx(temp_doc_path, new_docx_path)
                temp_doc_path = new_docx_path
            pdf_path = temp_doc_path.replace(f".{file_type}", ".pdf")
            docx_to_pdf(temp_doc_path, pdf_path)
            validate_pdf(pdf_path)
            images = convert_pdf_to_images(pdf_path)
        
        elif file_type in ["png", "jpg", "jpeg"]:
            image = Image.open(uploaded_file).convert("RGB")
            images = [image]
        else:
            st.error("Unsupported file type.")
        
        if images:
            st.markdown("### Original Images")
            cols = st.columns(3)
            for idx, image in enumerate(images):
                with cols[idx % 3]:
                    st.image(image, caption=f"Page {idx + 1}", use_container_width=True)
            
            # Classify with LeViT
            with st.spinner("Classifying with LeViT..."):
                avg_probs = process_and_classify(images, processor, selected_model_path, selected_model_name)
                final_class = classify_with_threshold(avg_probs)
            st.success("LeViT Classification completed!")
            label = "Estácio Exam" if final_class == 0 else "Uniasselvi Exam" if final_class == 2 else "Other"
            st.write(f"**Final Class:** {label}")
            
            # Process YOLO for checkmarks and draw filtered boxes
            st.markdown("### Images with YOLO Bounding Boxes (Checkmarks)")
            bbox_images = []
            all_checkmark_confidences = []
            for image in images:
                image_bbox, checkmark_confidences = draw_yolo_bboxes(image, yolo, conf_threshold=0.5)
                bbox_images.append(image_bbox)
                all_checkmark_confidences.extend(checkmark_confidences)
            
            cols_bbox = st.columns(3)
            for idx, image_bbox in enumerate(bbox_images):
                with cols_bbox[idx % 3]:
                    st.image(image_bbox, caption=f"Page {idx + 1} with YOLO", use_container_width=True)
            
            # Display checkmark statistics
            if all_checkmark_confidences:
                avg_conf = np.mean(all_checkmark_confidences)
                count_checkmarks = len(all_checkmark_confidences)
                st.write(f"**Average Checkmark Confidence:** {avg_conf:.2f}")
                st.write(f"**Number of Checkmarks Detected:** {count_checkmarks}")
            else:
                st.write("No checkmarks detected.")
                    
    except Exception as e:
        st.error(f"Error processing the file: {e}")

st.markdown("---")
st.markdown('<p class="main-text">Developed by <a href="#">Lucas Meireles</a></p>', unsafe_allow_html=True)

