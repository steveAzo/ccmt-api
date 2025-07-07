#  Disease Detection API - Backend Development Guide

## Project Overview
This project is a **Crop Disease Detection API** built with `FastAPI` and powered by an **EfficientNet B2** model trained to classify 22 crop disease classes across maize, cassava, tomato, and cashew plants.

It is the foundation of a **mobile-based crop disease detection solution** intended to help farmers easily identify diseases from images.

---

## Current Status
The following components are already completed and working:
- Model training using `timm` EfficientNetB2.
- Trained model achieving ~91% validation accuracy.
- Model file hosted and auto-downloaded on API startup.
- Image preprocessing pipeline is in place.
- `/predict` endpoint deployed and working.
-  Model successfully running on Render.

---

## Tech Stack
- Python
- FastAPI
- PyTorch
- timm (EfficientNet models)
- TorchVision
- Uvicorn
- Render (Deployment)

---

## Project Structure

```

.
├── main.py               # FastAPI server with /predict endpoint
├── model.py              # Model loading and prediction logic
├── utils.py              # Image preprocessing functions
├── requirements.txt      # All required packages
├── models/               # Stores downloaded model file
└── README.md             # You are here

````

---

## ⚡ Current API Endpoint

### POST `/predict`
**Request:**  
- `multipart/form-data` with key: `file`
- Value: Image file (example: leaf image)

**Response:**
```json
{
  "label": "Maize - leaf spot",
  "confidence": 0.7533
}
````

**Possible Errors:**

* `500`: Invalid image format
* `500`: Model prediction failure

---

## Completed So Far

* [x] Model training using `timm` EfficientNetB2
* [x] Model saved and hosted on GitHub releases
* [x] Backend auto-downloads model on first run
* [x] `/predict` endpoint built and tested
* [x] Model correctly deployed and serving on Render

---

## Backend Team - Next Steps

### 1. **User Authentication**

* Implement user management:

  * `POST /register`
  * `POST /login`
* Use JWT tokens for authentication.
* Store user profiles in PostgreSQL or another database.

---

### 2. **Prediction History Storage**

* Create a `PredictionHistory` table to store:

  * User ID
  * Prediction label
  * Confidence score
  * Timestamp
* Build endpoints for:

  * Retrieving user prediction history.
  * Deleting specific predictions if needed.

---

### 3. **Optional: Image Upload Storage**

* Consider storing user-uploaded images using:

  * AWS S3
  * Cloudinary
* Save the image URLs in the database for later retrieval.

---

### 4. **LLM Integration (Optional but Impressive)**

* After getting the prediction:

  * Send the disease label to an LLM (OpenAI, Gemini, etc.)
  * Retrieve additional information like:

    * Symptoms
    * Prevention
    * Recommended pesticides
* You can:

  * Call external LLM APIs.
  * Or mock this response locally if API keys are a blocker.

---

### 5. **Localization & Accessibility**

* Implement:

  * Local language translations (Twi, Ewe, Dagbani, etc.)
  * Audio feedback (text-to-speech API)
* Build this as a service that takes the LLM output and returns:

  * Local language text
  * Optional audio files

---

### 6. **Mobile App API Integration**

* Provide mobile devs with:

  * Authentication APIs
  * Predict API
  * History APIs
* Ensure CORS is properly configured.

---

### 7. **(Optional) Admin Dashboard**

* Build a simple dashboard to:

  * View predictions across users
  * Track app usage
  * Monitor potential abuse or issues

---

## Notes

* All API keys (LLM, storage, etc.) should be safely managed via environment variables.
* Proper error handling and input validation are key for stability.
* Feel free to improve the UI/UX of API responses for the mobile devs.




