from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os

app = FastAPI(
    title="Durian Disease Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "durian_disease_model.onnx")

CLASS_NAMES = [
    "Leaf_Algal",
    "Leaf_Blight",
    "Leaf_Colletotrichum",
    "Leaf_Healthy",
    "Leaf_Phomopsis",
    "Leaf_Rhizoctonia",
    "anthracnose_disease",
    "canker_disease",
    "fruit_rot",
    "mealybug_infestation",
    "pink_disease",
    "sooty_mold",
    "stem_blight",
    "stem_cracking_ gummosis",
    "thrips_disease",
    "yellow_leaf",
]

session = None


def get_session():
    global session

    if session is None:
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

    return session


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Durian Disease Detection API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_exists": os.path.exists(MODEL_PATH)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    x = np.asarray(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    model = get_session()
    input_name = model.get_inputs()[0].name

    prediction = model.run(None, {input_name: x})[0][0]

    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])

    return {
        "class_id": class_id,
        "disease": CLASS_NAMES[class_id],
        "confidence": confidence
    }
