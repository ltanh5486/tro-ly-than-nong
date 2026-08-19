"""
API nhận diện bệnh sầu riêng + chẩn đoán tổng hợp
+ hướng xử lý + thuốc/hoạt chất tham khảo.

AI ảnh chỉ là bước nhận diện ban đầu.
Hệ thống chẩn đoán sẽ tiếp tục thu thập thông tin
về triệu chứng, tưới, phân bón, thuốc BVTV và môi trường.
"""

from typing import Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from limiter import limiter

from ml.disease_inference import predict_disease
from ml.disease_treatments import get_disease_treatment
from ml.pesticide_registry import get_pesticide_recommendation
from ml.disease_diagnostic_engine import diagnose_disease


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


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def _optional_float(value: Optional[str]):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Optional[str]):
    """
    Đọc số nguyên từ câu trả lời tự nhiên.

    Ví dụ:
    - "10" -> 10
    - "10 ngày" -> 10
    - "khoảng 10 ngày" -> 10
    - "được 7 ngày rồi" -> 7
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    # Nếu người dùng nhập trực tiếp một số
    try:
        return int(float(value))
    except (TypeError, ValueError):
        pass

    # Nếu người dùng trả lời bằng câu tự nhiên
    import re

    match = re.search(
        r"-?\d+(?:[.,]\d+)?",
        value
    )

    if not match:
        return None

    number_text = (
        match.group(0)
        .replace(",", ".")
    )

    try:
        return int(float(number_text))
    except (TypeError, ValueError):
        return None


# ============================================================
# API NHẬN DIỆN + CHẨN ĐOÁN BAN ĐẦU
# ============================================================

@router.post(
    "/disease/predict",
    summary="Nhận diện và chẩn đoán bệnh sầu riêng từ ảnh",
)
@limiter.limit("30/minute")
async def disease_predict(
    request: Request,

    image: UploadFile = File(...),

    plant_part: str = Form("unknown"),
    symptoms: str = Form(""),
    location: str = Form(""),

    symptom_duration_days: str = Form(""),

    irrigation_method: str = Form(""),
    irrigation_amount: str = Form(""),
    irrigation_frequency: str = Form(""),

    fertilizer_name: str = Form(""),
    fertilizer_amount: str = Form(""),
    fertilizer_days_ago: str = Form(""),

    pesticide_name: str = Form(""),
    pesticide_amount: str = Form(""),
    pesticide_days_ago: str = Form(""),
    pesticide_mixed: str = Form(""),

    temperature: str = Form(""),
    humidity: str = Form(""),
    rainfall: str = Form(""),
):
    """
    Nhận ảnh cây sầu riêng và thực hiện:

    1. AI nhận diện bệnh từ ảnh.
    2. Tra cứu kiến thức bệnh.
    3. Tra cứu nhóm thuốc/hoạt chất.
    4. Phân tích các yếu tố gây nhiễu:
       - thời tiết
       - tưới tiêu
       - phân bón
       - thuốc BVTV
    5. Xác định thông tin còn thiếu.
    6. Sinh câu hỏi tiếp theo cho chatbot.

    Không kết luận thuốc chỉ dựa vào ảnh.
    """

    # ========================================================
    # 1. KIỂM TRA FILE
    # ========================================================

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

    # ========================================================
    # 2. AI NHẬN DIỆN
    # ========================================================

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

    # ========================================================
    # 3. CLASS AI CHÍNH
    # ========================================================

    class_name = prediction.get("disease")

    if not class_name:
        raise HTTPException(
            status_code=500,
            detail="Model không trả về lớp bệnh hợp lệ.",
        )

    confidence = float(
        prediction.get("confidence", 0.0)
    )

    # ========================================================
    # 4. KNOWLEDGE BỆNH
    # ========================================================

    treatment = get_disease_treatment(
        class_name
    )

    # ========================================================
    # 5. REGISTRY THUỐC
    # ========================================================

    pesticide = get_pesticide_recommendation(
        class_name
    )

    # ========================================================
    # 6. CONTEXT CHẨN ĐOÁN
    # ========================================================

    diagnostic_context = {
        "plant_part": plant_part,
        "symptoms": symptoms,
        "location": location,

        "symptom_duration_days":
            _optional_int(symptom_duration_days),

        "irrigation_method":
            irrigation_method,

        "irrigation_amount":
            irrigation_amount,

        "irrigation_frequency":
            irrigation_frequency,

        "fertilizer_name":
            fertilizer_name,

        "fertilizer_amount":
            fertilizer_amount,

        "fertilizer_days_ago":
            _optional_int(fertilizer_days_ago),

    "pesticide_name":
        pesticide_name,

    "pesticide_amount":
        pesticide_amount,

    "pesticide_days_ago":
        _optional_int(pesticide_days_ago),

    "pesticide_mixed":
        pesticide_mixed,

    "temperature":
    _optional_float(temperature),

        "humidity":
            _optional_float(humidity),

        "rainfall":
            _optional_float(rainfall),
    }

    # ========================================================
    # 7. ENGINE CHẨN ĐOÁN TỔNG HỢP
    # ========================================================

    diagnostic = diagnose_disease(
        disease=class_name,
        confidence=confidence,
        context=diagnostic_context,
    )

    # ========================================================
    # 8. ĐÁNH GIÁ VIỆC HIỂN THỊ THUỐC
    # ========================================================

    if class_name == "Leaf_Healthy":

        treatment_status = "not_needed"

    elif diagnostic.get(
        "can_recommend_treatment"
    ):

        if pesticide.get("groups"):
            treatment_status = "reference_available"
        else:
            treatment_status = "need_confirmation"

    else:

        treatment_status = "need_confirmation"

    # ========================================================
    # 9. RESPONSE
    # ========================================================

    return {
        "status": "success",

        "crop": "Sầu riêng",

        "plant_part": plant_part,
        "symptoms": symptoms,
        "location": location,

        "image_filename": image.filename,

        # ----------------------------------------------
        # AI IMAGE
        # ----------------------------------------------

        "prediction": prediction,

        # ----------------------------------------------
        # KNOWLEDGE BỆNH
        # ----------------------------------------------

        "treatment": treatment,

        # ----------------------------------------------
        # THUỐC / HOẠT CHẤT
        # ----------------------------------------------

        "pesticide": pesticide,

        "treatment_status":
            treatment_status,

        # ----------------------------------------------
        # ENGINE CHẨN ĐOÁN MỚI
        # ----------------------------------------------

        "diagnostic": diagnostic,

        # ----------------------------------------------
        # CHATBOT
        # ----------------------------------------------

        "diagnostic_chat": {

            "status":
                diagnostic.get("status"),

            "need_more_information":
                diagnostic.get("status")
                == "need_more_information",

            "next_question":
                diagnostic.get("next_question"),

            "missing_information":
                diagnostic.get(
                    "missing_information",
                    [],
                ),

            "confounding_factors":
                diagnostic.get(
                    "confounding_factors",
                    [],
                ),

            "can_recommend_treatment":
                diagnostic.get(
                    "can_recommend_treatment",
                    False,
                ),
        },

        # ----------------------------------------------
        # CONTEXT CHO CHATBOT / RAG
        # ----------------------------------------------

        "disease_context": {

            "class_name":
                class_name,

            "vi_name":
                treatment.get(
                    "vi_name",
                    class_name,
                ),

            "confidence":
                confidence,

            "plant_part":
                plant_part,

            "symptoms":
                symptoms,

            "location":
                location,

            "category":
                treatment.get(
                    "category"
                ),

            "likely_cause":
                treatment.get(
                    "likely_cause"
                ),

            "management":
                treatment.get(
                    "management",
                    [],
                ),

            "pesticide_groups":
                pesticide.get(
                    "groups",
                    [],
                ),

            "pesticide_note":
                pesticide.get(
                    "note",
                    "",
                ),

            # Context mới
            "diagnostic_inputs":
                diagnostic_context,

            "diagnostic_status":
                diagnostic.get(
                    "status"
                ),

            "diagnostic_confidence":
                diagnostic.get(
                    "diagnostic_confidence"
                ),

            "confounding_factors":
                diagnostic.get(
                    "confounding_factors",
                    [],
                ),
        },
    }

# ============================================================
# API CHẨN ĐOÁN ĐỐI THOẠI
# ============================================================

@router.post(
    "/disease/diagnose",
    summary="Chẩn đoán tổng hợp bệnh sầu riêng theo hội thoại",
)
@limiter.limit("60/minute")
async def disease_diagnose(
    request: Request,

    class_name: str = Form(...),
    confidence: float = Form(...),

    symptoms: str = Form(""),
    symptom_duration_days: str = Form(""),

    irrigation_method: str = Form(""),
    irrigation_amount: str = Form(""),
    irrigation_frequency: str = Form(""),

    fertilizer_name: str = Form(""),
    fertilizer_amount: str = Form(""),
    fertilizer_days_ago: str = Form(""),

    pesticide_name: str = Form(""),
    pesticide_amount: str = Form(""),
    pesticide_days_ago: str = Form(""),
    pesticide_mixed: str = Form(""),

    location: str = Form(""),

    temperature: str = Form(""),
    humidity: str = Form(""),
    rainfall: str = Form(""),
):
    """
    Nhận dữ liệu từng bước từ chatbot.

    Không cần gửi lại ảnh.
    Sử dụng kết quả AI trước đó + dữ liệu thực địa
    để quyết định cần hỏi thêm hay đã đủ thông tin.
    """

    context = {
        "symptoms":
            symptoms,

        "symptom_duration_days":
            _optional_int(symptom_duration_days),

        "irrigation_method":
            irrigation_method,

        "irrigation_amount":
            irrigation_amount,

        "irrigation_frequency":
            irrigation_frequency,

        "fertilizer_name":
            fertilizer_name,

        "fertilizer_amount":
            fertilizer_amount,

        "fertilizer_days_ago":
            _optional_int(fertilizer_days_ago),

        "pesticide_name":
            pesticide_name,

        "pesticide_amount":
            pesticide_amount,

        "pesticide_days_ago":
            _optional_int(pesticide_days_ago),

        "pesticide_mixed":
            pesticide_mixed,

        "location":
            location,

        "temperature":
            _optional_float(temperature),

        "humidity":
            _optional_float(humidity),

        "rainfall":
            _optional_float(rainfall),
    }

    diagnostic = diagnose_disease(
        disease=class_name,
        confidence=confidence,
        context=context,
    )

    # ========================================================
    # CÒN THIẾU THÔNG TIN → CHAT HỎI TIẾP
    # ========================================================

    if diagnostic.get("status") == "need_more_information":

        return {
            "status": "need_more_information",

            "next_question":
                diagnostic.get("next_question"),

            "missing_information":
                diagnostic.get(
                    "missing_information",
                    [],
                ),

            "diagnostic_confidence":
                diagnostic.get(
                    "diagnostic_confidence",
                ),

            "confounding_factors":
                diagnostic.get(
                    "confounding_factors",
                    [],
                ),
        }

    # ========================================================
    # ĐÃ ĐỦ THÔNG TIN
    # ========================================================

    treatment = get_disease_treatment(
        class_name
    )

    pesticide = get_pesticide_recommendation(
        class_name
    )

    vi_name = treatment.get(
        "vi_name",
        class_name,
    )

    management = treatment.get(
        "management",
        [],
    )

    pesticide_note = pesticide.get(
        "note",
        "",
    )

    confounders = diagnostic.get(
        "confounding_factors",
        [],
    )

    # Tạo phần giải thích ngắn cho chatbot
    factor_text = ""

    if confounders:
        factor_text = (
            "\n\n**Các yếu tố cần lưu ý:**\n" +
            "\n".join(
                f"- {item.get('message', '')}"
                for item in confounders
                if item.get("message")
            )
        )

    management_text = ""

    if management:
        management_text = (
            "\n\n**Khuyến nghị chăm sóc:**\n" +
            "\n".join(
                f"- {item}"
                for item in management
            )
        )

    pesticide_text = ""

    if pesticide_note:
        pesticide_text = (
            f"\n\n**Thuốc/hoạt chất:** {pesticide_note}"
        )

    answer = (
        f"🩺 **Chẩn đoán tổng hợp:** {vi_name}\n\n"
        f"Mức tin cậy sau khi kết hợp thông tin thực địa: "
        f"**{diagnostic.get('diagnostic_confidence', {}).get('level', 'chưa xác định')}**."
        f"{factor_text}"
        f"{management_text}"
        f"{pesticide_text}"
    )

    return {
        "status":
            diagnostic.get(
                "status",
                "need_field_confirmation",
            ),

        "answer":
            answer,

        "diagnosis":
            vi_name,

        "management":
            management,

        "pesticide_note":
            pesticide_note,

        "diagnostic_confidence":
            diagnostic.get(
                "diagnostic_confidence",
            ),

        "confounding_factors":
            confounders,

        "can_recommend_treatment":
            diagnostic.get(
                "can_recommend_treatment",
                False,
            ),
    }