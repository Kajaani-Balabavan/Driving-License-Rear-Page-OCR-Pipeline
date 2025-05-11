import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from paddleocr import PaddleOCR
from tqdm import tqdm

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Known vehicle classes
VEHICLE_CLASSES = {'A1', 'A', 'B1', 'B', 'C1', 'C', 'CE', 'D1', 'D', 'DE', 'G1', 'G', 'J'}

def format_date_string(text):
    """Format potential date strings by adding correct separators"""
    cleaned = re.sub(r'[./-]', '', text)
    if re.match(r'^\d{8}$', cleaned):
        return f"{cleaned[:2]}.{cleaned[2:4]}.{cleaned[4:]}"
    elif re.match(r'^\d{2}[./-]\d{6}$', text):
        parts = re.split(r'[./-]', text)
        return f"{parts[0]}.{parts[1][:2]}.{parts[1][2:]}"
    elif re.match(r'^\d{4}[./-]\d{4}$', text):
        parts = re.split(r'[./-]', text)
        return f"{parts[0][:2]}.{parts[0][2:]}.{parts[1]}"
    return text

def best_orientation_image(image):
    """Rotate image in 0, 90, 180, 270 and pick the one with best vehicle/date detection."""
    best_score = -1
    best_img = image
    best_angle = 0
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = image.copy()
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result = ocr.ocr(rotated, cls=True)
        score = 0
        for line in result:
            for word in line:
                if not word or len(word) < 2:
                    continue
                text = word[1][0]
                conf = word[1][1]
                if text in VEHICLE_CLASSES or re.match(r'\d{2}[./-]\d{2}[./-]\d{4}', text):
                    score += conf * 2
                else:
                    score += conf
        if score > best_score:
            best_score = score
            best_img = rotated
            best_angle = angle
    print(f"Selected orientation: {best_angle}° (score: {best_score:.2f})")
    return best_img, best_angle

def preprocess_image(image_path):
    """
    Enhanced preprocessing with rotation correction, contrast enhancement, and noise removal.
    Returns preprocessed image along with visualization of preprocessing steps.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Normalize orientation
    normalized, orientation = best_orientation_image(img)
    # Convert to grayscale
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    # Visualize all 4 stages in 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows x 2 columns
    # Row 1: Original and Normalized
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original')
    axs[0, 1].imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title(f'Normalized ({orientation}°)')
    # Row 2: Enhanced and Denoised (both grayscale)
    axs[1, 0].imshow(enhanced, cmap='gray')
    axs[1, 0].set_title('CLAHE Enhanced')
    axs[1, 1].imshow(denoised, cmap='gray')
    axs[1, 1].set_title('Denoised')
    # Format layout
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return denoised, normalized, orientation

def extract_ocr_data(image):
    """Extract OCR data with bounding box, confidence, and position information."""
    result = ocr.ocr(image, cls=True)
    data = []
    if not result:
        return data
    for line in result:
        if not line:
            continue
        for word in line:
            if not word or len(word) < 2:
                continue
            try:
                bbox = word[0]
                text_info = word[1]
                if not text_info or len(text_info) < 2:
                    continue
                text = text_info[0].strip()
                conf = float(text_info[1])
                if text:
                    x_min = min(point[0] for point in bbox)
                    y_min = min(point[1] for point in bbox)
                    x_max = max(point[0] for point in bbox)
                    y_max = max(point[1] for point in bbox)
                    data.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                        'center': ((x_min + x_max) / 2, (y_min + y_max) / 2)
                    })
            except Exception as e:
                print(f"Error processing OCR result: {e}")
                continue
    return data

def filter_potential_dates(ocr_data):
    """Filter and format potential dates from OCR data"""
    filtered_data = []
    for item in ocr_data:
        text = item['text']
        original_text = text
        if re.match(r'\d{2}[./-]\d{2}[./-]\d{4}', text):
            text = re.sub(r'[/-]', '.', text)
        elif re.match(r'^\d{8}$', text):
            text = format_date_string(text)
        elif re.match(r'\d{2,4}[./-]\d{4,6}', text):
            text = format_date_string(text)
        if re.match(r'\d{2}\.\d{2}\.\d{4}', text):
            new_item = item.copy()
            new_item['text'] = text
            new_item['original_text'] = original_text
            filtered_data.append(new_item)
        elif text in VEHICLE_CLASSES:
            filtered_data.append(item)
    return filtered_data

def visualize_annotations(img, ocr_data, orientation):
    """Draw OCR annotations with color coding by type"""
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    for item in ocr_data:
        bbox = item['bbox']
        text = item['text']
        color = (0, 255, 0) if text in VEHICLE_CLASSES else ((255, 0, 0) if re.match(r'\d{2}\.\d{2}\.\d{4}', text) else (0, 0, 255))
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(vis, f"{text} ({item['confidence']:.2f})", (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'OCR Annotations (Orientation: {orientation}°)')
    plt.axis('off')
    plt.show()
    return vis

def match_vehicles_with_dates(ocr_data, orientation):
    """Match vehicle classes with corresponding dates based on spatial proximity."""
    vehicles = [item for item in ocr_data if item['text'] in VEHICLE_CLASSES]
    dates = [item for item in ocr_data if re.match(r'\d{2}\.\d{2}\.\d{4}', text)]
    rows = {}
    for item in ocr_data:
        row_key = int(item['center'][1] // 20)
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append(item)
    matches = []
    for vehicle in vehicles:
        v_row_key = int(vehicle['center'][1] // 20)
        if v_row_key in rows:
            row_dates = [item for item in rows[v_row_key]
                         if re.match(r'\d{2}\.\d{2}\.\d{4}', item['text'])]
            if orientation in [0, 90]:
                row_dates.sort(key=lambda x: x['center'][0])
            else:
                row_dates.sort(key=lambda x: x['center'][0], reverse=True)
            start_date = row_dates[0]['text'] if len(row_dates) >= 1 else '-'
            expiry_date = row_dates[1]['text'] if len(row_dates) >= 2 else '-'
            start_original = row_dates[0].get('original_text', start_date) if len(row_dates) >= 1 else '-'
            expiry_original = row_dates[1].get('original_text', expiry_date) if len(row_dates) >= 2 else '-'
            matches.append({
                'Vehicle Class': vehicle['text'],
                'Start Date': start_date,
                'Expiry Date': expiry_date,
                'Start Date Original': start_original,
                'Expiry Date Original': expiry_original,
                'Confidence': vehicle['confidence']
            })
    detected_classes = {m['Vehicle Class'] for m in matches}
    for missing in VEHICLE_CLASSES - detected_classes:
        matches.append({
            'Vehicle Class': missing,
            'Start Date': '-',
            'Expiry Date': '-',
            'Start Date Original': '-',
            'Expiry Date Original': '-',
            'Confidence': 0.0
        })
    return matches

def process_license_image(image_path):
    """Full license processing pipeline with date correction and visualization."""
    try:
        processed_img, normalized_img, orientation = preprocess_image(image_path)
        ocr_data = extract_ocr_data(processed_img)
        if not ocr_data:
            print(f"No text detected in image: {image_path}")
            return pd.DataFrame()
        print("\nRaw OCR Results:")
        for idx, item in enumerate(ocr_data, 1):
            print(f"{idx}. [{item['confidence']:.2f}] {item['text']}")
            print(f"    Bounding Box: {item['bbox']}")
        filtered_data = filter_potential_dates(ocr_data)
        annotated_img = visualize_annotations(normalized_img, filtered_data, orientation)
        matches = match_vehicles_with_dates(filtered_data, orientation)
        result_df = pd.DataFrame(matches)
        print("\nParsed License Data:")
        display_df = result_df[['Vehicle Class', 'Start Date', 'Expiry Date', 'Confidence']]
        print(display_df.to_markdown(index=False))
        corrections = result_df[(result_df['Start Date'] != result_df['Start Date Original']) |
                               (result_df['Expiry Date'] != result_df['Expiry Date Original'])]
        if not corrections.empty:
            print("\nDate Corrections Applied:")
            for _, row in corrections.iterrows():
                if row['Start Date'] != row['Start Date Original'] and row['Start Date Original'] != '-':
                    print(f"  Class {row['Vehicle Class']} Start: {row['Start Date Original']} -> {row['Start Date']}")
                if row['Expiry Date'] != row['Expiry Date Original'] and row['Expiry Date Original'] != '-':
                    print(f"  Class {row['Vehicle Class']} Expiry: {row['Expiry Date Original']} -> {row['Expiry Date']}")
        return result_df
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_directory(input_dir, output_csv=None):
    """Process all images in a directory and save results to CSV."""
    all_results = []
    image_files = [f for f in os.listdir(input_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in tqdm(image_files, desc="Processing license images"):
        image_path = os.path.join(input_dir, filename)
        print(f"\n{'=' * 50}\nProcessing: {filename}\n{'=' * 50}")
        df = process_license_image(image_path)
        if not df.empty:
            df['Source File'] = filename
            all_results.append(df)
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        if output_csv:
            output_df = final_df.drop(['Start Date Original', 'Expiry Date Original', 'Confidence'], axis=1)
            output_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")
        return final_df
    return pd.DataFrame()

if __name__ == "__main__":
    input_dir = "./images"  # Change to your directory
    output_csv = "license_data.csv"
    results = process_directory(input_dir, output_csv)
    print("\nProcessing Complete!")
    print(f"Total valid entries: {len(results)}")
