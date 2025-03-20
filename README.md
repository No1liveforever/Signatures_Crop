# ✂️ Signatures Crop  
**Signatures_Crop** is a tool designed to detect and crop handwritten signatures from images. Using image processing techniques, it isolates signatures from documents, making them ready for further analysis or authentication.  

## 🛠 Features  
✅ Automatically detects and extracts signatures from scanned documents  
✅ Supports multiple image formats (JPG, PNG, etc.)  
✅ Uses OpenCV for efficient image processing  
✅ Can be integrated with signature verification systems  

## 📌 Technologies Used  
- **Python** 🐍  
- **OpenCV** 🎯 (for image processing)  
- **YOLOv5** 🚀 (for signature detection)  

## 🚀 How to Run  

### 1️⃣ Clone the Repository  

git clone https://github.com/No1liveforever/Signatures_Crop.git
cd Signatures_Crop


### 2️⃣ Install Dependencies  
Make sure you have Python installed, then install the required libraries:  

pip install -r requirements.txt


### 3️⃣ Run Signature Detection  
To crop signatures from an image, use:  

python detect_signatures.py --source your_image.jpg
Replace `your_image.jpg` with the path to your image.

### 4️⃣ Running on Webcam  
To detect signatures in real-time from a webcam:  

python webcam.py

### 5️⃣ Output  
- The cropped signature images will be saved in the **`runs/`** directory.  


## 🎯 Applications  
🔹 Digital document processing  
🔹 Automated signature verification  
🔹 Banking and legal document management  
