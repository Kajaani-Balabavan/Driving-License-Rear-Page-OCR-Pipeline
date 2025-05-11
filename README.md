# License Plate OCR Processor

This project uses PaddleOCR to extract vehicle class information and associated validity dates from **the backside of license**. It supports automatic orientation correction, date formatting, and structured output in CSV format.

## 🔧 Features

- Automatic image orientation correction using OCR feedback
- Text cleaning and date normalization
- Visualization of OCR detection areas
- Matching of vehicle classes with corresponding start/expiration dates
- Export to CSV

## 📦 Dependencies

Install required packages via pip:

```bash
pip install paddlepaddle paddleocr opencv-python numpy pandas matplotlib tqdm
```

## 📁 Folder Structure

Place your images in a folder named `images`, or change the path in the script accordingly.

## 🚀 Usage

Run the script directly:

```bash
python license_ocr_processor.py
```

Make sure you have an `images/` folder in the root directory containing `.jpg`, `.jpeg`, or `.png` files.

The results will be saved in `license_data.csv`.

## 📁 Output Example

| Vehicle Class | Start Date | Expiry Date | Source File |
|---------------|------------|-------------|-------------|
| C             | 01.01.2022 | 01.01.2030  | license1.jpg |

## 📝 Notes

- This tool assumes licenses are printed in a consistent layout where each vehicle class appears near its two dates.
- Date formats like `DDMMYYYY`, `DD/MM/YYYY`, etc., are automatically corrected.
