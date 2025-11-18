# AI Text-to-Image Generator (Stable Diffusion)

This project is my implementation of the ML Internship task for Talrn.  
It demonstrates a working text-to-image generation system using Stable Diffusion v1.5,  
running through a Streamlit web interface with GPU acceleration on Google Colab.

---

## Features

- Generate images from text prompts
- Adjustable parameters:
  - Style preset
  - Negative prompt
  - CFG scale
  - Sampling steps
  - Number of output images
- Download generated PNG images
- Clean Streamlit UI
- GPU support (Google Colab T4)

---

## Sample Outputs

See the `outputs/` folder for examples generated using the app.

---

##  Setup & Installation

### 1. Clone the repository
git clone https://github.com/AntaraP741/text2image

cd text2image

### 2. Install dependencies
pip install -r requirements.txt


### 3. Run the app
streamlit run app.py


For GPU-supported execution, open the project in **Google Colab**,  
install the same requirements, and run the Streamlit server.

---

## Model Used

- Stable Diffusion v1.5  
- HuggingFace Diffusers Pipeline  
- DPMSolverMultistepScheduler  

---

## Ethical Usage

This project is educational.  
AI-generated images must be labeled to avoid confusion with real imagery.  
Inappropriate or harmful prompt generation should be avoided.

---

## Contact

Antara Prasad  
Email: antaraprasad2017@gmail.com
