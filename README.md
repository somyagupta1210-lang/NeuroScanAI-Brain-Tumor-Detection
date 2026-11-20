# Brain Tumor Detection - AI-Based MRI Analysis and Reporting System

## 1. Backend
    cd backend
    python -m venv venv
    venv\Scripts\activate          (Windows)   |   source venv/bin/activate  (Mac/Linux)
    pip install -r requirements.txt
    python app.py                  -> http://localhost:5000

## 2. Frontend (new terminal)
    cd frontend
    npm install
    npm run dev                    -> http://localhost:5173

## 3. Train the real model
Put the Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar) into dataset/Training and dataset/Testing, then:
    python train_model.py
This creates model/brain_tumor_model.h5 and model/metrics.json. Restart the backend.
Until then the app runs in DEMO mode (placeholder scores, clearly labelled).

## 4. Sample MRIs
Copy 3 images from dataset/Testing into backend/samples/.

Research prototype only - not a medical diagnosis.
