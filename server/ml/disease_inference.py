"""ONNX inference cho mô hình nhận diện bệnh sầu riêng."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image


MODEL_PATH = os.getenv(
    "DURIAN_DISEASE_MODEL",
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "ai_models",
        "durian_disease.onnx",
    ),
)

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

# Hai tensor phụ mà model ONNX hiện tại yêu cầu.
# Lấy trực tiếp từ lớp Normalization của model Keras gốc.
NORMALIZATION_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 1, 3)

NORMALIZATION_VARIANCE = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 1, 3)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


@lru_cache(maxsize=1)
def _get_session():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Chưa tìm thấy model nhận diện bệnh: {MODEL_PATH}"
        )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu onnxruntime. Hãy cài dependency trong requirements.txt."
        ) from exc

    return ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )


def _prepare_image(
    image_bytes: bytes,
    input_shape: list[Any],
) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    array = np.asarray(image, dtype=np.float32)

    # Hỗ trợ cả model NCHW và NHWC.
    if len(input_shape) == 4 and input_shape[1] == 3:
        array = np.transpose(array, (2, 0, 1))

    return np.expand_dims(array, axis=0).astype(np.float32)


def predict_disease(
    image_bytes: bytes,
    top_k: int = 3,
) -> dict[str, Any]:
    session = _get_session()

    inputs = session.get_inputs()
    input_meta = inputs[0]

    tensor = _prepare_image(
        image_bytes,
        input_meta.shape,
    )

    # Model ONNX hiện tại có 3 input:
    # 1. Ảnh
    # 2. Mean của lớp Normalization
    # 3. Variance của lớp Normalization
    feed = {
        inputs[0].name: tensor,
    }

    if len(inputs) >= 3:
        feed[inputs[1].name] = NORMALIZATION_MEAN
        feed[inputs[2].name] = NORMALIZATION_VARIANCE

    output = session.run(
        None,
        feed,
    )[0]

    output = np.asarray(output).squeeze()

    if output.ndim != 1:
        raise RuntimeError(
            f"Output model không hợp lệ: shape={output.shape}"
        )

    if output.shape[0] != len(CLASS_NAMES):
        raise RuntimeError(
            "Số lớp output của model không khớp CLASS_NAMES: "
            f"{output.shape[0]} != {len(CLASS_NAMES)}"
        )

    probabilities = output.astype(np.float64)

    # Một số model ONNX xuất logits thay vì xác suất.
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(
            probabilities.sum(),
            1.0,
            atol=1e-3,
        )
    ):
        probabilities = _softmax(probabilities)

    top_k = max(
        1,
        min(top_k, len(CLASS_NAMES)),
    )

    indices = np.argsort(probabilities)[::-1][:top_k]

    predictions = [
        {
            "class": CLASS_NAMES[int(index)],
            "confidence": round(
                float(probabilities[index]),
                4,
            ),
        }
        for index in indices
    ]

    best = predictions[0]
    confidence = best["confidence"]

    if confidence >= 0.80:
        confidence_level = "cao"
    elif confidence >= 0.60:
        confidence_level = "trung bình"
    else:
        confidence_level = "thấp"

    return {
        "disease": best["class"],
        "confidence": confidence,
        "confidence_level": confidence_level,
        "top_predictions": predictions,
        "disclaimer": (
            "Kết quả nhận diện từ ảnh chỉ mang tính hỗ trợ. "
            "Cần kết hợp triệu chứng, bộ phận cây, điều kiện "
            "môi trường và kiểm tra thực địa."
        ),
    }