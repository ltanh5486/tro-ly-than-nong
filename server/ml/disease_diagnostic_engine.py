"""
disease_diagnostic_engine.py

Bộ máy chẩn đoán tổng hợp bệnh sầu riêng.

Nguyên tắc:
- Không kết luận chỉ dựa vào ảnh.
- Kết hợp AI ảnh với dữ liệu thời tiết, tưới tiêu,
  phân bón, thuốc BVTV và triệu chứng thực địa.
- Chatbot hỏi lần lượt các thông tin còn thiếu.
- Phát hiện yếu tố có thể gây nhiễu kết quả AI.
- Chỉ cho phép khuyến nghị điều trị khi có đủ dữ liệu cơ bản.
"""

from typing import Any, Dict, List, Optional


# ============================================================
# CÁC TRƯỜNG CHẨN ĐOÁN
# ============================================================

DIAGNOSTIC_FIELDS = {
    # Triệu chứng
    "symptoms": "Triệu chứng thực tế trên cây",
    "symptom_duration_days": "Triệu chứng xuất hiện bao nhiêu ngày",
    "plant_part": "Bộ phận cây bị ảnh hưởng",

    # Tưới
    "irrigation_method": "Phương pháp tưới",
    "irrigation_amount": "Lượng nước tưới",
    "irrigation_frequency": "Tần suất tưới",

    # Phân bón
    "fertilizer_name": "Loại phân bón gần nhất",
    "fertilizer_amount": "Lượng phân bón",
    "fertilizer_days_ago": "Thời gian từ lần bón gần nhất",

    # Thuốc BVTV
    "pesticide_name": "Tên thuốc hoặc hoạt chất",
    "pesticide_amount": "Liều/nồng độ thuốc đã sử dụng",
    "pesticide_days_ago": "Thời gian từ lần phun gần nhất",
    "pesticide_mixed": "Có pha chung nhiều thuốc hay không",

    # Môi trường
    "temperature": "Nhiệt độ",
    "humidity": "Độ ẩm",
    "rainfall": "Lượng mưa",
    "location": "Vùng trồng",
}


# ============================================================
# CÂU HỎI CHATBOT
# ============================================================

QUESTION_MAP = {
    "symptoms":
        "Ngoài biểu hiện trong ảnh, cây còn có triệu chứng gì khác không?",

    "symptom_duration_days":
        "Triệu chứng này xuất hiện khoảng bao nhiêu ngày rồi?",

    "plant_part":
        "Triệu chứng xuất hiện ở lá, cành, thân, rễ, trái hay nhiều bộ phận?",

    "irrigation_method":
        "Vườn đang tưới theo cách nào: nhỏ giọt, phun mưa, tưới gốc hay cách khác?",

    "irrigation_amount":
        "Mỗi lần tưới khoảng bao nhiêu lít/cây hoặc bao nhiêu m³/ha?",

    "irrigation_frequency":
        "Bao lâu vườn được tưới một lần?",

    "fertilizer_name":
        "Gần đây có bón phân không? Nếu có, đang dùng loại phân nào? Nếu không có thì trả lời 'không'.",

    "fertilizer_amount":
        "Lần bón gần nhất dùng khoảng bao nhiêu kg/cây hoặc kg/ha?",

    "fertilizer_days_ago":
        "Lần bón phân gần nhất cách đây khoảng bao nhiêu ngày?",

    "pesticide_name":
        "Gần đây có phun thuốc trừ sâu, thuốc nấm hoặc thuốc BVTV không? Nếu có, cho biết tên thuốc hoặc hoạt chất; nếu không có thì trả lời 'không'.",

    "pesticide_amount":
        "Thuốc được pha với liều hoặc nồng độ bao nhiêu? Có thể cho biết ml/bình, g/bình hoặc theo liều trên nhãn.",

    "pesticide_days_ago":
        "Lần phun thuốc gần nhất cách đây khoảng bao nhiêu ngày?",

    "pesticide_mixed":
        "Lần phun gần nhất có pha chung nhiều loại thuốc hoặc pha chung với phân bón lá không?",

    "location":
        "Vườn sầu riêng nằm ở khu vực nào?",

    "temperature":
        "Nhiệt độ tại vườn hiện khoảng bao nhiêu °C?",

    "humidity":
        "Độ ẩm tại vườn hiện khoảng bao nhiêu %?",

    "rainfall":
        "Gần đây khu vực vườn có mưa nhiều hoặc mưa liên tục không?",
}


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def _has_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    return True


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _is_negative_answer(value: Any) -> bool:
    """
    Nhận biết người dùng nói rằng KHÔNG sử dụng
    phân bón hoặc thuốc BVTV.
    """

    text = _normalize_text(value)

    negative_answers = {
        "không",
        "khong",
        "ko",
        "k",
        "không có",
        "khong co",
        "chưa",
        "chua",
        "chưa dùng",
        "chua dung",
        "không dùng",
        "khong dung",
        "không phun",
        "khong phun",
        "không bón",
        "khong bon",
        "none",
        "no",
    }

    return text in negative_answers

def _is_positive_answer(value: Any) -> bool:
    """
    Nhận biết câu trả lời chỉ xác nhận CÓ,
    nhưng chưa cung cấp tên phân bón hoặc thuốc BVTV.
    """

    text = _normalize_text(value)

    positive_answers = {
        "có",
        "co",
        "c",
        "yes",
        "y",
        "có dùng",
        "co dung",
        "có bón",
        "co bon",
        "có phun",
        "co phun",
        "có sử dụng",
        "co su dung",
    }

    return text in positive_answers

# ============================================================
# XÁC ĐỊNH THÔNG TIN CÒN THIẾU
# ============================================================

def get_missing_information(
    context: Dict[str, Any]
) -> List[str]:

    missing: List[str] = []

    # ========================================================
    # 1. TRIỆU CHỨNG + TƯỚI TIÊU
    # ========================================================

    basic_fields = [
        "symptoms",
        "symptom_duration_days",
        "irrigation_method",
        "irrigation_amount",
        "irrigation_frequency",
    ]

    for field in basic_fields:
        if not _has_value(context.get(field)):
            missing.append(field)

    # ========================================================
    # 2. PHÂN BÓN
    # ========================================================

    fertilizer_name = context.get("fertilizer_name")

    # Chưa trả lời có bón phân hay không
    if not _has_value(fertilizer_name):
        missing.append("fertilizer_name")

    # Nếu trả lời "có" nhưng chưa nói TÊN PHÂN
    elif _is_positive_answer(fertilizer_name):
        missing.append("fertilizer_name")

    # Nếu trả lời "không" -> bỏ qua lượng và thời gian bón
    elif _is_negative_answer(fertilizer_name):
        pass

    # Có tên phân cụ thể -> hỏi tiếp lượng + thời gian
    else:
        if not _has_value(
            context.get("fertilizer_amount")
        ):
            missing.append("fertilizer_amount")

        if not _has_value(
            context.get("fertilizer_days_ago")
        ):
            missing.append("fertilizer_days_ago")

    # ========================================================
    # 3. THUỐC BVTV
    # ========================================================

    pesticide_name = context.get("pesticide_name")

    # Chưa trả lời có dùng thuốc hay không
    if not _has_value(pesticide_name):
        missing.append("pesticide_name")

    # Trả lời "có" nhưng chưa nói TÊN THUỐC/HOẠT CHẤT
    elif _is_positive_answer(pesticide_name):
        missing.append("pesticide_name")

    # Trả lời "không" -> bỏ qua các câu hỏi chi tiết về thuốc
    elif _is_negative_answer(pesticide_name):
        pass

    # Có tên thuốc cụ thể -> hỏi tiếp liều, thời gian, pha chung
    else:
        if not _has_value(
            context.get("pesticide_amount")
        ):
            missing.append("pesticide_amount")

        if not _has_value(
            context.get("pesticide_days_ago")
        ):
            missing.append("pesticide_days_ago")

        if not _has_value(
            context.get("pesticide_mixed")
        ):
            missing.append("pesticide_mixed")

    return missing


# ============================================================
# CÂU HỎI TIẾP THEO
# ============================================================

def get_next_question(
    context: Dict[str, Any]
) -> Optional[Dict[str, str]]:

    missing = get_missing_information(
        context
    )

    if not missing:
        return None

    field = missing[0]

    # ========================================================
    # PHÂN BÓN:
    # Đã trả lời "có" nhưng chưa nói loại phân
    # ========================================================

    if (
        field == "fertilizer_name"
        and _is_positive_answer(
            context.get("fertilizer_name")
        )
    ):
        return {
            "field": "fertilizer_name",
            "question":
                "Bà con đang sử dụng loại phân nào? "
                "Ví dụ NPK 16-16-8, NPK 20-20-15, "
                "phân hữu cơ hoặc loại khác.",
        }

    # ========================================================
    # THUỐC BVTV:
    # Đã trả lời "có" nhưng chưa nói tên thuốc
    # ========================================================

    if (
        field == "pesticide_name"
        and _is_positive_answer(
            context.get("pesticide_name")
        )
    ):
        return {
            "field": "pesticide_name",
            "question":
                "Bà con đã sử dụng thuốc hoặc hoạt chất nào gần nhất? "
                "Nếu nhớ tên sản phẩm hoặc hoạt chất thì cho biết tên.",
        }

    # ========================================================
    # CÂU HỎI THÔNG THƯỜNG
    # ========================================================

    return {
        "field": field,
        "question": QUESTION_MAP[field],
    }


# ============================================================
# PHÁT HIỆN YẾU TỐ GÂY NHIỄU
# ============================================================

def detect_confounding_factors(
    context: Dict[str, Any]
) -> List[Dict[str, str]]:

    factors: List[Dict[str, str]] = []

    # ========================================================
    # THUỐC BVTV
    # ========================================================

    pesticide_name = context.get(
        "pesticide_name"
    )

    pesticide_days = _to_int(
        context.get(
            "pesticide_days_ago"
        )
    )

    pesticide_amount = context.get(
        "pesticide_amount"
    )

    pesticide_mixed = _normalize_text(
        context.get(
            "pesticide_mixed"
        )
    )

    if (
        _has_value(pesticide_name)
        and not _is_negative_answer(
            pesticide_name
        )
    ):

        if (
            pesticide_days is not None
            and pesticide_days <= 7
        ):

            factors.append({
                "factor": "recent_pesticide",
                "level": "high",
                "message":
                    "Cây vừa được sử dụng thuốc BVTV trong 7 ngày gần đây. "
                    "Triệu chứng trên lá có thể bị thay đổi hoặc có khả năng "
                    "liên quan đến phản ứng sau phun thuốc."
            })

        else:

            factors.append({
                "factor": "pesticide_history",
                "level": "medium",
                "message":
                    "Có lịch sử sử dụng thuốc BVTV. Cần đối chiếu thời điểm "
                    "xuất hiện triệu chứng với thời điểm phun thuốc."
            })

        if _has_value(pesticide_amount):

            factors.append({
                "factor": "pesticide_dose_available",
                "level": "info",
                "message":
                    "Đã có thông tin về liều hoặc nồng độ thuốc. "
                    "Cần đối chiếu với hướng dẫn trên nhãn sản phẩm."
            })

        mixed_keywords = [
            "có",
            "co",
            "có pha",
            "co pha",
            "pha chung",
            "trộn",
            "tron",
            "yes",
        ]

        if any(
            keyword == pesticide_mixed
            or keyword in pesticide_mixed
            for keyword in mixed_keywords
        ):

            factors.append({
                "factor": "pesticide_mixture",
                "level": "high",
                "message":
                    "Có pha chung nhiều sản phẩm. Đây là yếu tố cần xem xét "
                    "vì hỗn hợp thuốc hoặc thuốc với phân bón lá có thể làm "
                    "tăng nguy cơ phản ứng bất lợi trên cây."
            })

    # ========================================================
    # PHÂN BÓN
    # ========================================================

    fertilizer_name = context.get(
        "fertilizer_name"
    )

    fertilizer_amount = context.get(
        "fertilizer_amount"
    )

    fertilizer_days = _to_int(
        context.get(
            "fertilizer_days_ago"
        )
    )

    if (
        _has_value(fertilizer_name)
        and not _is_negative_answer(
            fertilizer_name
        )
    ):

        if (
            fertilizer_days is not None
            and fertilizer_days <= 7
        ):

            factors.append({
                "factor": "recent_fertilizer",
                "level": "medium",
                "message":
                    "Cây vừa được bón phân trong 7 ngày gần đây. "
                    "Cần xem xét khả năng stress dinh dưỡng, nồng độ muối "
                    "hoặc tổn thương rễ nếu triệu chứng xuất hiện sau bón."
            })

        if _has_value(
            fertilizer_amount
        ):

            factors.append({
                "factor": "fertilizer_amount_available",
                "level": "info",
                "message":
                    "Đã có lượng phân sử dụng. Cần đối chiếu với tuổi cây, "
                    "giai đoạn sinh trưởng và khuyến cáo của loại phân."
            })

    # ========================================================
    # ĐỘ ẨM
    # ========================================================

    humidity = _to_float(
        context.get("humidity")
    )

    if humidity is not None:

        if humidity >= 85:

            factors.append({
                "factor": "high_humidity",
                "level": "high",
                "message":
                    "Độ ẩm môi trường cao, tạo điều kiện thuận lợi "
                    "cho nhiều nhóm nấm bệnh phát triển."
            })

        elif humidity <= 40:

            factors.append({
                "factor": "low_humidity",
                "level": "medium",
                "message":
                    "Độ ẩm thấp có thể gây stress mất nước, làm khô "
                    "hoặc cháy mép lá."
            })

    # ========================================================
    # NHIỆT ĐỘ
    # ========================================================

    temperature = _to_float(
        context.get("temperature")
    )

    if (
        temperature is not None
        and temperature >= 34
    ):

        factors.append({
            "factor": "high_temperature",
            "level": "medium",
            "message":
                "Nhiệt độ cao có thể gây stress nhiệt hoặc làm "
                "triệu chứng trên lá biểu hiện mạnh hơn."
        })

    # ========================================================
    # MƯA
    # ========================================================

    rainfall = _to_float(
        context.get("rainfall")
    )

    if (
        rainfall is not None
        and rainfall > 30
    ):

        factors.append({
            "factor": "heavy_rain",
            "level": "high",
            "message":
                "Lượng mưa cao làm tăng độ ẩm đất, nguy cơ úng rễ "
                "và tạo điều kiện cho một số tác nhân nấm phát triển."
        })

    # ========================================================
    # TƯỚI
    # ========================================================

    irrigation_method = _normalize_text(
        context.get(
            "irrigation_method"
        )
    )

    irrigation_frequency = _normalize_text(
        context.get(
            "irrigation_frequency"
        )
    )

    if "phun" in irrigation_method:

        factors.append({
            "factor": "overhead_irrigation",
            "level": "medium",
            "message":
                "Tưới phun có thể làm lá ẩm kéo dài, tạo điều kiện "
                "cho một số bệnh trên lá phát triển."
        })

    frequent_keywords = [
        "hàng ngày",
        "mỗi ngày",
        "1 ngày",
        "ngày nào cũng",
    ]

    if any(
        keyword in irrigation_frequency
        for keyword in frequent_keywords
    ):

        factors.append({
            "factor": "frequent_irrigation",
            "level": "medium",
            "message":
                "Tần suất tưới cao. Cần kết hợp lượng nước và khả năng "
                "thoát nước để đánh giá nguy cơ dư nước hoặc úng rễ."
        })

    return factors


# ============================================================
# ĐỘ TIN CẬY CHẨN ĐOÁN TỔNG HỢP
# ============================================================

def calculate_diagnostic_confidence(
    ai_confidence: float,
    context: Dict[str, Any]
) -> Dict[str, Any]:

    score = float(
        ai_confidence or 0
    )

    missing = get_missing_information(
        context
    )

    confounders = detect_confounding_factors(
        context
    )

    # Thiếu thông tin -> chưa nên tin hoàn toàn kết quả ảnh
    score -= min(
        len(missing) * 0.02,
        0.20
    )

    high_count = sum(
        1
        for item in confounders
        if item.get("level") == "high"
    )

    medium_count = sum(
        1
        for item in confounders
        if item.get("level") == "medium"
    )

    score -= high_count * 0.04
    score -= medium_count * 0.02

    score = max(
        0.0,
        min(score, 1.0)
    )

    if score >= 0.80:
        level = "cao"

    elif score >= 0.60:
        level = "trung bình"

    else:
        level = "thấp"

    return {
        "score": round(
            score,
            4
        ),

        "level": level,

        "missing_information":
            missing,

        "confounding_factors":
            confounders,
    }


# ============================================================
# CHẨN ĐOÁN TỔNG HỢP
# ============================================================

def build_integrated_diagnosis(
    prediction: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:

    disease = prediction.get(
        "disease",
        "unknown"
    )

    ai_confidence = float(
        prediction.get(
            "confidence",
            0
        ) or 0
    )

    diagnostic = (
        calculate_diagnostic_confidence(
            ai_confidence,
            context
        )
    )

    missing = diagnostic[
        "missing_information"
    ]

    confounders = diagnostic[
        "confounding_factors"
    ]

    # ========================================================
    # TRẠNG THÁI
    # ========================================================

    if missing:

        status = (
            "need_more_information"
        )

        next_question = (
            get_next_question(
                context
            )
        )

    else:

        next_question = None

        if diagnostic["score"] >= 0.60:

            status = (
                "ready_for_recommendation"
            )

        else:

            status = (
                "need_field_confirmation"
            )

    # ========================================================
    # GIẢI THÍCH
    # ========================================================

    reasoning: List[str] = []

    reasoning.append(
        f"Mô hình ảnh nhận diện lớp '{disease}' "
        f"với độ tin cậy "
        f"{ai_confidence * 100:.1f}%."
    )

    if confounders:

        reasoning.append(
            "Có yếu tố chăm sóc hoặc môi trường "
            "có thể ảnh hưởng đến biểu hiện triệu chứng."
        )

    if missing:

        reasoning.append(
            "Chưa đủ thông tin thực địa để đưa ra "
            "khuyến nghị chăm sóc hoặc thuốc cuối cùng."
        )

    else:

        reasoning.append(
            "Đã thu thập đủ nhóm thông tin chăm sóc "
            "cơ bản để thực hiện đánh giá tổng hợp."
        )

    return {
        "status": status,

        "image_prediction": {
            "disease": disease,
            "confidence": ai_confidence,
        },

        "diagnostic_confidence": {
            "score":
                diagnostic["score"],

            "level":
                diagnostic["level"],
        },

        "missing_information":
            missing,

        "confounding_factors":
            confounders,

        "reasoning":
            reasoning,

        "next_question":
            next_question,

        "can_recommend_treatment":
            status
            == "ready_for_recommendation",
    }


# ============================================================
# HÀM CHÍNH
# ============================================================

def diagnose_disease(
    disease: str,
    confidence: float,
    context: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    if context is None:
        context = {}

    prediction = {
        "disease": disease,
        "confidence": confidence,
    }

    return build_integrated_diagnosis(
        prediction,
        context
    )