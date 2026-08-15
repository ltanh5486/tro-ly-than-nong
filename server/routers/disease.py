from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

import models
from limiter import limiter
from ml.disease_inference import predict_disease
from routers.auth import get_current_user


router = APIRouter(
    prefix="/api",
    tags=["Disease Detection"],
)


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/disease/predict",
    summary="Nhận diện bệnh sầu riêng từ ảnh",
)
@limiter.limit("30/minute")
async def disease_predict(
    request: Request,
    image: UploadFile = File(...),
    plant_part: str = Form("unknown"),
    symptoms: str = Form(""),
    location: str = Form(""),
    ):
    """
    Nhận ảnh cây sầu riêng và trả về Top-3 lớp bệnh dự đoán.

    Kết quả từ mô hình ảnh chỉ mang tính hỗ trợ, không phải chẩn đoán
    cuối cùng. Các trường plant_part, symptoms và location được giữ lại
    để phục vụ bước tích hợp Decision Engine/RAG tiếp theo.
    """

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Chỉ hỗ trợ ảnh JPG, JPEG, PNG hoặc WEBP.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Tệp ảnh rỗng.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Ảnh vượt quá dung lượng tối đa 10 MB.",
        )

    try:
        prediction = predict_disease(
            image_bytes=image_bytes,
            top_k=3,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Không thể xử lý ảnh: {exc}",
        ) from exc

    return {
        "status": "success",
        "crop": "Sầu riêng",
        "plant_part": plant_part,
        "symptoms": symptoms,
        "location": location,
        "image_filename": image.filename,
        "prediction": prediction,
    }
