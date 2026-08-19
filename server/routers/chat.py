import os
import re
import logging
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from openai import AsyncOpenAI
from langchain_core.output_parsers import StrOutputParser

from schemas import ChatRequest, ChatResponse, ChatMessage
from routers.auth import get_current_user
import database, models
from ml.rag_engine import rag_engine
from ml.decision_engine import run_decision_engine
from ml.inference import predict_risk, predict_price, get_weather, get_latest_price
from ml.expert_rules import run_expert_check
from config import LOCATION_MAPPING

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

load_dotenv()

logger = logging.getLogger(__name__)
from limiter import limiter

router = APIRouter(
    prefix="/api",
    tags=["Chat"],
)

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_llm = None
if GEMINI_API_KEY and ChatGoogleGenerativeAI:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        max_output_tokens=2048,
        max_retries=1,
        timeout=15
    )

# LM Studio Configuration
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "google/gemma-2-9b")
API_KEY_VAL = os.getenv("OPENROUTER_API_KEY", "lm-studio")

llm_client = None
if LM_STUDIO_URL:
    llm_client = AsyncOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=API_KEY_VAL,
    )

_GREETING_WORDS = ["xin chào", "chào bạn", "chào", "hello", "hi", "hey", "alo"]
_THANKS_WORDS = ["cảm ơn", "cám ơn", "thanks", "thank you", "tạm biệt", "bye"]
_FINANCE_KEYWORDS = ["chi phí", "giá bao nhiêu", "tốn bao nhiêu", "đầu tư", "vốn", "lợi nhuận", "doanh thu", "roi"]

def classify_intent(message: str) -> str:
    """
    Chỉ nhận diện ý định được nói RÕ trong câu hiện tại.
    Không dùng lịch sử hội thoại ở hàm này để tránh kéo nhầm chủ đề cũ.
    """
    msg = message.lower().strip()

    if any(re.search(rf"\b{re.escape(g)}\b", msg) for g in _GREETING_WORDS):
        return "greeting"

    if any(re.search(rf"\b{re.escape(t)}\b", msg) for t in _THANKS_WORDS):
        return "thanks"

    if msg in {"ok", "oke", "okay", "ừ", "uh", "uhm", "được", "được rồi", "hiểu rồi"}:
        return "thanks"

    price_list_keywords = [
        "biết giá cây gì", "có giá cây gì", "có giá những cây nào",
        "có giá nông sản nào", "hệ thống có giá gì", "hệ thống có giá cây nào",
        "bạn có giá gì", "giá được cây nào", "giá được những cây nào",
        "đang có giá cây nào", "biết giá loại nào", "có giá loại nào",
        "có giá những loại nào", "đang có giá loại nào", "giá những loại nào",
    ]
    if any(k in msg for k in price_list_keywords):
        return "price_list"

    # Quyết định bán phải được ưu tiên trước price/agriculture.
    # "giá bán" vẫn là câu hỏi giá; còn mọi câu có động từ bán là market.
    if "bán" in msg and "giá bán" not in msg:
        return "market"

    price_keywords = [
        "giá hôm nay", "giá hiện tại", "giá sầu riêng", "giá cà phê",
        "giá chè", "giá trà", "giá nông sản", "bao nhiêu một kg",
        "bao nhiêu/kg", "giá bán", "giá bao nhiêu", "giá thế nào", "giá sao",
        "hỏi giá", "xem giá",
    ]
    if any(k in msg for k in price_keywords):
        return "price"

    # Bệnh cây / xử lý bệnh: đây là chuyển chủ đề rõ ràng, không được
    # để lịch sử giá kéo ngược sang intent "price".
    disease_keywords = [
        "bệnh cây", "cây bị bệnh", "bị bệnh", "xử lý bệnh",
        "trị bệnh", "phòng bệnh", "chữa bệnh", "bệnh như thế",
        "bệnh đó", "bệnh này", "sâu bệnh", "nấm bệnh"
    ]
    if any(k in msg for k in disease_keywords):
        return "disease"

    weather_keywords = [
        "thời tiết", "nhiệt độ", "mưa", "độ ẩm", "nắng", "dự báo thời tiết",
        "vùng trồng", "đất ở", "phù hợp trồng", "trồng ở đâu",
    ]
    if any(k in msg for k in weather_keywords):
        return "weather"

    market_keywords = [
        "xu hướng giá", "phân tích xu hướng", "dự báo giá",
        "có nên bán", "nên bán", "bán luôn", "bán bây giờ",
        "giữ lại", "găm hàng", "chốt lời", "bán hay giữ",
    ]
    if any(k in msg for k in market_keywords):
        return "market"

    finance_keywords = [
        "chi phí", "tốn bao nhiêu", "đầu tư", "vốn", "lợi nhuận",
        "doanh thu", "roi", "hòa vốn", "hiệu quả kinh tế",
    ]
    if any(k in msg for k in finance_keywords):
        return "finance"

    return "agriculture"


def _has_active_disease_context(body: ChatRequest) -> bool:
    return bool(
        body.context
        and (
            getattr(body.context, "disease_context", None)
            or getattr(body.context, "disease_session", None)
        )
    )


def _looks_like_disease_reply(message: str) -> bool:
    msg = message.lower().strip()
    terms = [
        "chảy mủ", "không chảy mủ", "ko chảy mủ", "vàng lá", "rụng lá",
        "đốm", "thối", "nứt", "héo", "cháy lá", "rễ", "thân", "cành",
        "lá", "trái", "tưới", "bón", "phân", "thuốc", "nhỏ giọt",
        "phun mưa", "tưới gốc", "ngày", "tuần"
    ]
    return any(term in msg for term in terms)



def _history_pairs(history) -> List[Dict[str, str]]:
    """Chuẩn hóa history thành role/content để state machine không phụ thuộc kiểu Pydantic."""
    out = []
    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role", "") or "").lower()
            content = str(item.get("content", "") or "")
        else:
            role = str(getattr(item, "role", "") or "").lower()
            content = str(getattr(item, "content", "") or "")
        if content.strip():
            out.append({"role": role, "content": content.strip()})
    return out


def _explicit_topic(message: str) -> Optional[str]:
    """Chỉ trả topic khi câu hiện tại nói RÕ chủ đề mới."""
    intent = classify_intent(message)
    if intent != "agriculture":
        return intent

    msg = message.lower().strip()

    if any(k in msg for k in [
        "vùng này có thể trồng cây gì", "vùng này trồng cây gì",
        "phù hợp trồng cây gì", "nên trồng cây gì", "cây gì phù hợp",
        "cây trồng phù hợp", "vùng này phù hợp"
    ]):
        return "suitability"

    cultivation_terms = [
        "chăm sóc", "tưới", "tưới tiêu", "bón phân", "phân bón",
        "cắt tỉa", "làm cỏ", "trồng thế nào", "kỹ thuật trồng",
        "thu hoạch thế nào", "nuôi trái", "làm bông"
    ]
    disease_treatment_terms = [
        "bệnh cây", "bệnh như thế", "xử lý bệnh", "trị bệnh",
        "dùng thuốc thế nào", "phun thuốc thế nào", "thuốc thế nào"
    ]
    if any(k in msg for k in disease_treatment_terms):
        return "disease"

    if any(k in msg for k in cultivation_terms):
        return "cultivation"
    return None


def _pending_from_assistant(text: str) -> Optional[str]:
    """Nhận diện dữ liệu mà câu hỏi cuối của trợ lý đang chờ người dùng trả lời."""
    t = (text or "").lower()

    if ("bao nhiêu ngày" in t or "mấy ngày" in t) and any(
        k in t for k in ["trái", "quả", "sầu riêng", "tuổi"]
    ):
        return "fruit_age"

    if any(k in t for k in [
        "cây gì", "loại cây gì", "trồng cây gì", "mẫu lá", "cây nhà"
    ]):
        return "crop"

    if any(k in t for k in ["bao nhiêu tuổi", "mấy tuổi", "cây bao nhiêu năm"]):
        return "plant_age"

    if any(k in t for k in ["giai đoạn nào", "đang mang trái", "đang ra hoa"]):
        return "growth_stage"

    if any(k in t for k in ["triệu chứng", "chảy mủ", "vàng lá", "rụng lá"]):
        return "symptom"

    if any(k in t for k in ["đã phun", "thuốc nào", "loại thuốc"]):
        return "pesticide"

    if any(k in t for k in ["đã bón", "phân nào", "loại phân"]):
        return "fertilizer"

    return None


def _topic_from_text(text: str) -> Optional[str]:
    """Đọc topic từ một lượt hội thoại đã có, ưu tiên tín hiệu rõ."""
    t = (text or "").lower()

    if any(k in t for k in [
        "có nên bán", "nên bán", "bán lúc này", "bán luôn",
        "xu hướng giá", "phân tích xu hướng", "dự báo giá",
        "chốt lời", "bán hay giữ", "giữ lại"
    ]):
        return "market"

    if any(k in t for k in [
        "bệnh cây", "cây bị bệnh", "phomopsis", "chảy mủ",
        "vàng lá", "rụng lá", "nấm bệnh", "xử lý bệnh", "trị bệnh"
    ]):
        return "disease"

    if any(k in t for k in [
        "giá hiện tại", "vnđ/kg", "vnd/kg", "giá nông sản",
        "giá hôm nay", "xem giá"
    ]):
        return "price"

    if any(k in t for k in [
        "thời tiết", "nhiệt độ", "độ ẩm", "lượng mưa",
        "phù hợp trồng cây gì", "vùng này phù hợp"
    ]):
        return "weather"

    if any(k in t for k in [
        "chăm sóc", "tưới tiêu", "tưới nước", "bón phân",
        "cắt tỉa", "kỹ thuật trồng"
    ]):
        return "cultivation"

    return None



def _disease_crop_from_context(body: ChatRequest) -> Optional[str]:
    """
    Crop của ca bệnh là một namespace riêng.
    Tuyệt đối không lấy crop từ market/history giá để ghi đè ca bệnh ảnh.
    """
    ctx = getattr(body, "context", None)
    if ctx is None:
        return None

    disease_text = " ".join([
        str(getattr(ctx, "disease_context", "") or ""),
        str(getattr(ctx, "disease_session", "") or ""),
    ])
    return detect_crop_in_message(disease_text)


def _market_crop_from_history(history) -> Optional[str]:
    """Crop gần nhất chỉ trong mạch giá/bán."""
    items = _history_pairs(history)
    market_crop = None
    market_active = False

    for item in items[-12:]:
        t = item["content"]
        topic = _topic_from_text(t)
        crop = detect_crop_in_message(t)

        if topic in {"price", "market"}:
            market_active = True
            if crop:
                market_crop = crop
        elif topic in {"disease", "weather", "cultivation"}:
            market_active = False

        # Câu crop ngắn ngay sau mạch market/price.
        if market_active and crop:
            market_crop = crop

    return market_crop


def get_conversation_state(body: ChatRequest) -> Dict[str, Any]:
    """
    State tách namespace:
      market.crop  = cây/nông sản đang hỏi giá/bán
      disease.crop = cây của ca bệnh ảnh
    Không cho hai crop ghi đè lẫn nhau.
    """
    items = _history_pairs(getattr(body, "history", None))

    state: Dict[str, Any] = {
        "topic": None,
        "crop": None,              # crop hiệu lực của topic hiện tại
        "market_crop": _market_crop_from_history(getattr(body, "history", None)),
        "disease_crop": _disease_crop_from_context(body),
        "pending": None,
        "last_assistant": "",
    }

    # UI crop chỉ dùng như fallback chung, KHÔNG ghi đè disease crop đã có.
    ctx = getattr(body, "context", None)
    ui_crop = getattr(ctx, "crop", None) if ctx is not None else None

    for item in items[-12:]:
        content = item["content"]
        topic = _topic_from_text(content)
        if topic:
            state["topic"] = topic

        if item["role"] == "assistant":
            state["last_assistant"] = content
            pending = _pending_from_assistant(content)
            if pending:
                state["pending"] = pending
            elif "?" in content:
                state["pending"] = None

    if state["topic"] in {"price", "market"}:
        state["crop"] = state["market_crop"] or ui_crop
    elif state["topic"] == "disease":
        state["crop"] = state["disease_crop"] or ui_crop
    else:
        state["crop"] = ui_crop

    return state


def _is_short_slot_answer(message: str, pending: Optional[str]) -> bool:
    """Câu ngắn như '3 tháng', 'không', 'cà phê' được xem là trả lời slot đang chờ."""
    if not pending:
        return False
    msg = message.lower().strip()
    if not msg or len(msg) > 80:
        return False

    if pending in {"fruit_age", "plant_age"}:
        return bool(re.search(r"\b\d+(?:[.,]\d+)?\s*(?:ngày|tháng|năm|tuần)\b", msg))

    if pending == "crop":
        return bool(detect_crop_in_message(message))

    if pending in {"symptom", "pesticide", "fertilizer", "growth_stage"}:
        return len(msg.split()) <= 12

    return False


def _enrich_followup_query(message: str, state: Dict[str, Any], intent: str) -> str:
    """
    Biến câu phụ thuộc ngữ cảnh thành câu tự đủ nghĩa trước khi gửi LLM/RAG.
    Ví dụ '3 tháng' -> 'Sầu riêng Ri6; chủ đề quyết định bán; tuổi trái: 3 tháng'.
    """
    parts = []
    crop = state.get("crop")
    pending = state.get("pending")

    if crop:
        parts.append(f"Cây/nông sản đang nói tới: {crop}")

    topic_labels = {
        "market": "quyết định bán/giữ và thời điểm thu hoạch",
        "disease": "bệnh cây và xử lý",
        "cultivation": "chăm sóc canh tác",
        "weather": "điều kiện vùng trồng/thời tiết",
        "price": "giá nông sản",
    }
    if intent in topic_labels:
        parts.append(f"Chủ đề hiện tại: {topic_labels[intent]}")

    pending_labels = {
        "fruit_age": "Tuổi trái người dùng vừa trả lời",
        "plant_age": "Tuổi cây người dùng vừa trả lời",
        "crop": "Loại cây người dùng vừa trả lời",
        "growth_stage": "Giai đoạn sinh trưởng người dùng vừa trả lời",
        "symptom": "Triệu chứng người dùng vừa trả lời",
        "pesticide": "Thông tin thuốc người dùng vừa trả lời",
        "fertilizer": "Thông tin phân bón người dùng vừa trả lời",
    }
    if pending in pending_labels:
        parts.append(f"{pending_labels[pending]}: {message}")
    else:
        parts.append(f"Câu hỏi hiện tại: {message}")

    return ". ".join(parts)


def infer_followup_intent(message: str, history) -> Optional[str]:
    """Suy luận câu nối tiếp theo chủ đề gần nhất."""
    if not history:
        return None

    msg = message.lower().strip()
    if len(msg) > 120:
        return None

    previous_text = " ".join(
        (getattr(item, "content", "") or "").lower()
        for item in history[-2:]
    )
    crop = detect_crop_in_message(message)

    generic_followup = any(k in msg for k in [
        "thì sao", "còn", "thế nào", "vậy", "loại này",
        "có nên", "nên không", "nên ko", "bán không", "bán ko"
    ])

    market_terms = [
        "xu hướng giá", "phân tích xu hướng", "dự báo giá", "thị trường",
        "có nên bán", "nên bán", "bán luôn", "bán hay giữ",
        "giữ lại", "găm hàng", "chốt lời"
    ]
    price_terms = [
        "vnđ/kg", "vnd/kg", "giá hiện tại", "dữ liệu giá", "xem giá",
        "hỏi giá", "giá nông sản", "giá loại nào", "giá cây gì",
        "cung cấp giá", "giá hôm nay"
    ]

    if any(k in previous_text for k in market_terms):
        if any(k in msg for k in ["bán", "giữ", "xu hướng", "giá", "chốt lời"]) or generic_followup:
            return "market"

    if any(k in previous_text for k in price_terms):
        if any(k in msg for k in ["biết loại nào", "có loại nào", "những loại nào", "loại nào"]):
            return "price_list"
        if any(k in msg for k in ["bán", "giữ", "xu hướng", "dự báo", "chốt lời"]):
            return "market"
        if crop or generic_followup:
            return "price"

    disease_terms = [
        "bệnh", "phomopsis", "nấm", "chảy mủ", "vàng lá",
        "rụng lá", "đốm", "thối", "héo", "cháy lá"
    ]
    if any(k in previous_text for k in disease_terms):
        if _looks_like_disease_reply(message) or any(k in msg for k in [
            "xử lý", "trị", "phòng", "bệnh", "thuốc", "bón", "tưới", "chăm sóc"
        ]):
            return "disease"

    weather_terms = ["thời tiết", "nhiệt độ", "lượng mưa", "độ ẩm"]
    if any(k in previous_text for k in weather_terms):
        if crop or any(k in msg for k in ["ngày mai", "hôm nay", "tuần tới", "thì sao", "còn"]):
            return "weather"

    return None


def resolve_intent(message: str, body: ChatRequest) -> str:
    """
    State machine:
    1) Chủ đề được nói rõ trong câu hiện tại luôn thắng.
    2) Nếu trợ lý đang chờ một slot và user trả lời ngắn, giữ nguyên topic.
    3) Nếu không, mới dùng suy luận history cũ làm fallback.
    """
    explicit = _explicit_topic(message)
    if explicit:
        return explicit

    state = get_conversation_state(body)

    if _is_short_slot_answer(message, state.get("pending")) and state.get("topic"):
        return state["topic"]

    followup = infer_followup_intent(message, body.history)
    if followup:
        return followup

    if _has_active_disease_context(body) and _looks_like_disease_reply(message):
        return "disease"

    return "agriculture"


def detect_crop_in_message(message: str) -> Optional[str]:
    msg = message.lower()
    # Ánh xạ từ khóa sang tên chuẩn trong hệ thống
    crop_keywords = {
        "sầu riêng": "Sầu riêng Ri6",
        "robusta": "Cà phê Robusta",
        "arabica": "Cà phê Arabica",
        "cà phê": "Cà phê Robusta", # Mặc định nếu chỉ nói cà phê
        "chè": "Chè Ô Long",
        "trà": "Chè Ô Long",
        "măng cụt": "Măng cụt",
        "bơ": "Bơ",
        "bở": "Bơ"
    }
    for kw, full_name in crop_keywords.items():
        if kw in msg:
            return full_name
    return None


def _extract_disease_name(disease_context=None, disease_session=None) -> str:
    candidates = []

    def collect(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            for key in ("class_name", "disease", "diagnosis"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip())
            nested = obj.get("final_diagnosis")
            if isinstance(nested, dict):
                collect(nested)
            return

        for key in ("class_name", "disease", "diagnosis"):
            val = getattr(obj, key, None)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())

        nested = getattr(obj, "final_diagnosis", None)
        if nested is not None:
            collect(nested)

    collect(disease_context)
    collect(disease_session)
    return candidates[0] if candidates else ""


def _is_disease_followup_query(query: str) -> bool:
    q = query.lower().strip()
    markers = [
        "bệnh cây", "bệnh như thế", "bệnh đó", "bệnh này",
        "xử lý sao", "xử lý thế nào", "trị sao", "trị thế nào",
        "phải làm gì", "chăm sóc thế nào", "có cần dùng thuốc"
    ]
    return any(m in q for m in markers)


def _build_retrieval_query(query: str, disease_context=None, disease_session=None) -> str:
    disease_name = _extract_disease_name(disease_context, disease_session)
    if disease_name and _is_disease_followup_query(query):
        return f"{disease_name}. Biện pháp xử lý, phòng trừ và chăm sóc. Câu hỏi: {query}"
    return query


def _disease_context_fallback(disease_context=None, disease_session=None) -> str:
    def as_dict(obj):
        if isinstance(obj, dict):
            return obj
        if obj is None:
            return {}
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        return {}

    for src in (disease_context, disease_session):
        data = as_dict(src)
        if not data:
            continue

        final_diag = data.get("final_diagnosis")
        if isinstance(final_diag, dict):
            answer = final_diag.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

            management = final_diag.get("management")
            diagnosis = final_diag.get("diagnosis") or data.get("class_name") or data.get("disease")
            if management:
                if isinstance(management, list):
                    items = "\n".join(f"- {x}" for x in management if str(x).strip())
                else:
                    items = f"- {management}"
                prefix = f"Với **{diagnosis}**, " if diagnosis else ""
                return f"{prefix}bạn nên ưu tiên:\n{items}"

        management = data.get("management")
        if management:
            if isinstance(management, list):
                items = "\n".join(f"- {x}" for x in management if str(x).strip())
            else:
                items = f"- {management}"
            return f"Bạn nên ưu tiên các biện pháp sau:\n{items}"

    return ""


async def _get_rag_response(
    query: str,
    history=None,
    disease_context=None,
    disease_session=None,
    weather_context=None
) -> str:
    # Filter nội dung nhạy cảm
    query_lower = query.lower()
    forbidden = ["quên hết", "chính trị", "tôn giáo", "mã nguồn", "password"]
    if any(k in query_lower for k in forbidden):
        return "Xin lỗi, tôi chỉ hỗ trợ các vấn đề về kỹ thuật canh tác nông nghiệp tại Lâm Đồng."


        # Bộ nhớ hội thoại và ca bệnh vừa nhận diện
    memory_parts = []

    if disease_context:
        memory_parts.append(
            f"THÔNG TIN CA BỆNH HIỆN TẠI:\n{disease_context}"
        )

    if disease_session:
        memory_parts.append(
            f"PHIÊN CHẨN ĐOÁN BỆNH:\n{disease_session}"
        )

    if history:
        memory_parts.append(
            f"LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n{history}"
        )

    if weather_context:
        memory_parts.append(
            f"THỜI TIẾT THỰC TẾ VÀ GẦN ĐÂY:\n{weather_context}"
        )

    memory_context = "\n\n".join(memory_parts)
    # Tăng top_k lên 5 để lấy ngữ cảnh đầy đủ hơn
    retrieval_query = _build_retrieval_query(
        query,
        disease_context=disease_context,
        disease_session=disease_session,
    )
    context = rag_engine.get_relevant_context(retrieval_query, top_k=5)
    
    # Nếu context quá yếu, AI vẫn nên cố gắng trả lời dựa trên kiến thức chung của nó
    # nhưng kèm theo cảnh báo là không tìm thấy trong tài liệu nội bộ.
    context_str = context if context else "Không có dữ liệu cụ thể trong sổ tay kỹ thuật."
    
    # Tối ưu context an toàn cho Gemini (Giới hạn 8000 ký tự - tương đương ~2000 tokens)
    if len(context_str) > 8000:
        context_str = context_str[:8000] + "\n...[Nội dung đã được cắt giảm]..."

    if gemini_llm:
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Bạn là 'Trợ Lý Thần Nông' - một chuyên gia nông nghiệp tận tâm tại vùng đất Lâm Đồng.\n"
                "PHONG CÁCH: Thân thiện, rõ ràng, tự nhiên. Luôn xưng hô với người dùng là 'bạn'. "
                "Không gọi người dùng là 'bà con' và không lặp lại lời chúc dài dòng sau mỗi câu trả lời.\n\n"
                "NHIỆM VỤ: Hỗ trợ người dùng về các vấn đề nông nghiệp như cây trồng, "
                "kỹ thuật canh tác, sâu bệnh, phân bón, tưới nước, đất đai, thời tiết, "
                "vùng trồng, giá nông sản, hiệu quả đầu tư và các vấn đề sản xuất nông nghiệp khác. "
                "Ưu tiên dữ liệu thực tế của hệ thống khi có; nếu hệ thống chưa có dữ liệu chuyên biệt "
                "cho một cây hoặc một vùng thì phải nói rõ giới hạn đó, không được tự tạo số liệu.\n\n"
                "BỘ NHỚ CA BỆNH VÀ HỘI THOẠI:\n{memory}\n\n"
                "Nếu BỘ NHỚ có ca bệnh hiện tại, phải hiểu các câu hỏi tiếp theo theo ca bệnh đó, trừ khi người dùng chuyển sang chủ đề khác.\n\n"
                "NGỮ CẢNH TỪ SỔ TAY KỸ THUẬT:\n{context}\n\n"
                "QUY TẮC PHẢN HỒI:\n"
                "1. Ưu tiên tuyệt đối dữ liệu trong NGỮ CẢNH và BỘ NHỚ CA BỆNH.\n"
                "2. Có thể dùng kiến thức nông nghiệp phổ thông để giải thích cơ chế, nguyên nhân và nguyên tắc chăm sóc, nhưng KHÔNG được tự tạo ra số liệu kỹ thuật cụ thể.\n"
                "3. TUYỆT ĐỐI KHÔNG tự suy đoán liều thuốc, nồng độ thuốc, lượng phân bón, lượng nước tưới, số lần tưới, thời gian cách ly hoặc lịch phun nếu những thông tin đó không có trong NGỮ CẢNH.\n"
                "4. Nếu người dùng hỏi một thông số kỹ thuật cụ thể mà dữ liệu hiện có chưa đủ để xác định an toàn, phải nói rõ chưa đủ dữ liệu và hỏi thêm thông tin cần thiết như tuổi cây, giai đoạn sinh trưởng, loại đất, thời tiết hoặc tên sản phẩm.\n"
                "5. Khi tư vấn thuốc BVTV, chỉ nêu hoạt chất/liều lượng khi có căn cứ trong dữ liệu kỹ thuật. Không tự bịa tên thương mại, liều lượng hoặc thời gian cách ly.\n"
                "6. Nếu BỘ NHỚ có ca bệnh hiện tại, phải sử dụng lại tên bệnh, kết quả AI, triệu chứng, tưới, phân bón và thuốc BVTV đã thu thập; KHÔNG hỏi lại thông tin người dùng đã cung cấp.\n"
                "7. Phân biệt rõ: dữ liệu từ ca bệnh, dữ liệu từ sổ tay kỹ thuật và kiến thức giải thích chung.\n"
                "8. Khi tư vấn tưới nước hoặc độ ẩm, phải ưu tiên xem dữ liệu thời tiết vài ngày gần đây và dự báo ngắn hạn trước khi hỏi người dùng về mưa/nắng.\n"
                "9. Không hỏi lại người dùng những thông tin thời tiết mà hệ thống đã có. Chỉ hỏi những yếu tố thực địa chưa biết như tuổi cây, loại đất, độ ẩm vùng rễ hoặc tình trạng thoát nước.\n"
                "10. Nếu mấy ngày gần đây đã có mưa đáng kể hoặc dự báo tiếp tục mưa, không được máy móc khuyên tăng tưới. Nếu nhiều ngày nóng, ít mưa thì mới xem xét nguy cơ thiếu ẩm, nhưng vẫn phải đối chiếu tình trạng đất thực tế.\n"
                "11. Trả lời trực tiếp câu hỏi hiện tại, không cần lặp lại toàn bộ chẩn đoán nếu người dùng chỉ hỏi tiếp một vấn đề cụ thể.\n"
                "12. KHÔNG trả lời các vấn đề ngoài phạm vi nông nghiệp.\n"
            )),
            ("human", "{query}")
        ])
        chain = prompt | gemini_llm | StrOutputParser()
        try:
            res = await chain.ainvoke({
    "query": query,
    "context": context_str,
    "memory": memory_context or "Chưa có ca bệnh hoặc hội thoại trước đó."
})
            if res:
                res = re.sub(r'<thought>.*?</thought>', '', res, flags=re.DOTALL)
                return res.strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg or "resourceexhausted" in error_msg:
                # Không khóa phiên chat. Gemini hết quota thì tự động chuyển sang LM Studio.
                logger.warning(f"Gemini Rate Limit hit - fallback to LM Studio: {e}")
            else:
                logger.error(f"Gemini API Error: {e}", exc_info=True)
            # Tiếp tục xuống LM Studio thay vì chặn người dùng.

    if llm_client:
        try:
            response = await llm_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Bạn là 'Trợ Lý Thần Nông' - chuyên gia nông nghiệp tại Lâm Đồng.\n"
                        f"NGỮ CẢNH HỖ TRỢ:\n{context_str}\n\n"
                        "QUY TẮC:\n"
                        "1. Ưu tiên thông tin trong NGỮ CẢNH.\n"
                        "2. TRẢ LỜI CỰC KỲ NGẮN GỌN, súc tích, tập trung vào giải pháp kỹ thuật.\n"
                        "3. Dùng gạch đầu dòng, tối đa 3-5 ý chính.\n"
                        "4. KHÔNG trả lời các vấn đề ngoài nông nghiệp."
                    )},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=1024,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-OpenRouter-Title": "Tro Ly Than Nong",
                }
            )
            res = response.choices[0].message.content
            if res:
                # Loại bỏ các thẻ suy nghĩ (thought) nếu model có trả về
                res = re.sub(r'<thought>.*?</thought>', '', res, flags=re.DOTALL)
                res = res.strip()
                return res
        except Exception as e:
            logger.error(f"Chat API Error: {e}", exc_info=True)

            disease_fallback = _disease_context_fallback(
                disease_context=disease_context,
                disease_session=disease_session,
            )
            if disease_fallback:
                return disease_fallback

            if not context:
                return (
                    "Hiện tôi chưa tìm thấy tài liệu kỹ thuật đủ phù hợp cho câu hỏi này. "
                    "Bạn hãy cho biết rõ tên cây và vấn đề cần hỏi để tôi tìm lại chính xác hơn."
                )

            return (
                "Tôi đã tìm thấy tài liệu liên quan nhưng dịch vụ tạo câu trả lời đang tạm gián đoạn. "
                "Bạn có thể gửi lại câu hỏi ngay; phiên hội thoại và ngữ cảnh hiện tại vẫn được giữ."
            )
    else:
        logger.error("No LLM client configured!")

        disease_fallback = _disease_context_fallback(
            disease_context=disease_context,
            disease_session=disease_session,
        )
        if disease_fallback:
            return disease_fallback

        if context:
            return (
                "Tôi đã tìm thấy tài liệu liên quan nhưng dịch vụ tạo câu trả lời đang tạm gián đoạn. "
                "Bạn có thể gửi lại câu hỏi ngay; phiên hội thoại và ngữ cảnh hiện tại vẫn được giữ."
            )
        return "Rất tiếc, hệ thống đang bảo trì phần tư vấn tự động."

async def _stream_rag_response(query: str):
    """Generator function to stream response from LLM."""
    context = rag_engine.get_relevant_context(query, top_k=5)
    context_str = context if context else "Không có dữ liệu cụ thể trong sổ tay kỹ thuật."
    
    if len(context_str) > 8000:
        context_str = context_str[:8000] + "\n..."

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Bạn là 'Trợ Lý Thần Nông' - Chuyên gia tư vấn nông nghiệp tại Lâm Đồng.\n"
            "Hãy trả lời bà con bằng giọng văn nhiệt tình, rõ ràng.\n\n"
            "DỮ LIỆU KỸ THUẬT:\n{context}\n\n"
            "QUY TẮC:\n"
            "1. Trình bày gạch đầu dòng rõ ràng, dễ đọc trên điện thoại.\n"
            "2. Tập trung vào giải pháp khắc phục thực tế.\n"
            "3. Nếu cần sử dụng thuốc BVTV, hãy nhắc bà con tuân thủ nguyên tắc 4 đúng.\n"
            "4. Kết hợp khéo léo thông tin từ dữ liệu kỹ thuật và kinh nghiệm thực tế tại địa phương."
        )),
        ("human", "{query}")
    ])

    if gemini_llm:
        try:
            chain = prompt | gemini_llm | StrOutputParser()
            async for chunk in chain.astream({"query": query, "context": context_str}):
                # Lọc bỏ thẻ thought nếu có
                clean_chunk = re.sub(r'<thought>.*?</thought>', '', chunk, flags=re.DOTALL)
                if clean_chunk:
                    yield clean_chunk
            return
        except Exception as e:
            logger.error(f"Gemini Streaming Error: {e}")
            # Nếu Gemini lỗi, fallback sang logic text bình thường hoặc yield lỗi
    
    if llm_client:
        try:
            response = await llm_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": f"Bạn là Trợ Lý Thần Nông. Ngữ cảnh: {context_str}"},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=0.3
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            return
        except Exception as e:
            logger.error(f"LLM Client Streaming Error: {e}")
            
    yield "⚠️ AI đang bận hoặc gặp lỗi kết nối. Vui lòng thử lại sau giây lát."

def _run_finance_simulation(crop: str, capital: float, area_ha: float, location: str, mode: str) -> str:
    loc_info = LOCATION_MAPPING.get(location, LOCATION_MAPPING["Phường B'Lao"])
    weather = get_weather(location)
    expert = run_expert_check(crop, location, loc_info["elevation"], weather["temp_max"], weather["temp_min"], weather["precipitation"])
    risk_level, _ = predict_risk(location, crop)
    cur_price = get_latest_price(crop)
    forecast_data = predict_price(crop, cur_price, location)
    pred_30d = forecast_data[-1]["predicted"]
    
    res = run_decision_engine(crop, risk_level, cur_price, pred_30d, capital, area_ha, weather["temp_min"], weather["precipitation"], mode, expert["ecology_violation"])
    fa = res["financial_analysis"]
    def fmt_money(v): return f"{v/1_000_000:.0f} triệu" if v < 1_000_000_000 else f"{v/1_000_000_000:.1f} tỷ"
    
    return (
        f"Phân tích tài chính cho **{crop}** tại **{location}** ({area_ha} ha):\n"
        f"• Chi phí đầu tư: {fmt_money(fa['estimated_cost'])}\n"
        f"• Doanh thu dự tính: {fmt_money(fa['estimated_revenue'])}\n"
        f"• Lợi nhuận dự kiến: {fmt_money(fa['estimated_profit'])}\n"
        f"• ROI: {fa['roi_pct']:.1f}%\n\n"
        f"**Khuyến nghị**: {res['production_decision']['recommendation']}"
    )

def _price_list_answer():
    answer = (
        "Hiện hệ thống có dữ liệu giá cho:\n"
        "- **Cà phê Robusta**\n"
        "- **Cà phê Arabica**\n"
        "- **Sầu riêng Ri6**\n"
        "- **Chè Ô Long**\n\n"
        "Bạn muốn xem giá loại nào?"
    )
    suggestions = [
        "Giá cà phê Robusta?",
        "Giá cà phê Arabica?",
        "Giá sầu riêng Ri6?",
        "Giá chè Ô Long?",
    ]
    return answer, suggestions


def _price_answer(message: str, body: ChatRequest):
    # Không dùng crop của ca bệnh cũ khi người dùng vừa chuyển sang hỏi giá.
    crop = detect_crop_in_message(message)

    if not crop:
        return (
            "Bạn muốn xem giá loại nông sản nào? Ví dụ: sầu riêng, cà phê Robusta, "
            "cà phê Arabica hoặc chè.",
            [
                "Giá sầu riêng hôm nay?",
                "Giá cà phê Robusta hôm nay?",
                "Giá cà phê Arabica hôm nay?",
                "Giá chè hôm nay?",
            ],
        )

    current_price = get_latest_price(crop)
    if current_price is None:
        return (
            f"Hiện hệ thống chưa có dữ liệu giá đủ tin cậy cho **{crop}** "
            f"để cung cấp giá hôm nay.",
            [],
        )

    return (
        f"Giá hiện tại hệ thống đang ghi nhận cho **{crop}** "
        f"là khoảng **{current_price:,.0f} VNĐ/kg**.",
        ["Giá 30 ngày tới thế nào?", "Có nên bán lúc này không?", "Phân tích xu hướng giá"],
    )



def _history_for_topic(history, intent: str):
    """Giữ một cửa sổ hội thoại nhỏ nhưng đủ để LLM hiểu câu hỏi nối tiếp."""
    if not history:
        return []
    items = list(history)

    if intent == "weather":
        return items[-2:]
    if intent == "market":
        return items[-6:]
    if intent == "disease":
        return items[-8:]
    if intent == "cultivation":
        return items[-6:]
    return items[-4:]



# ============================================================
# RUNTIME CONVERSATION SESSION
# ============================================================
# Tách hẳn các namespace để market crop không ghi đè disease crop.
# Key = user_id. Có endpoint reset để frontend gọi khi đóng chat.
_CHAT_SESSIONS: Dict[str, Dict[str, Any]] = {}
_CHAT_SESSION_TTL_SECONDS = 2 * 60 * 60


def _new_runtime_session() -> Dict[str, Any]:
    return {
        "active_topic": None,
        "market": {
            "crop": None,
            "action": None,             # sell / trend / price
            "fruit_age_text": None,     # ví dụ: "3 tháng"
            "harvest_in_text": None,    # ví dụ: "1 tháng nữa"
        },
        "disease": {
            "crop": None,
            "name": None,
            "active": False,
        },
        "cultivation": {
            "crop": None,
        },
        "pending": {
            "field": None,
            "topic": None,
            "crop": None,
        },
        "last_answer": "",
        "updated_at": time.time(),
    }


def _session_key(user_id) -> str:
    return str(user_id) if user_id is not None else "__anonymous__"


def _get_runtime_session(user_id, body: ChatRequest) -> Dict[str, Any]:
    key = _session_key(user_id)
    now = time.time()
    session = _CHAT_SESSIONS.get(key)

    if (
        session is None
        or now - float(session.get("updated_at", 0)) > _CHAT_SESSION_TTL_SECONDS
    ):
        session = _new_runtime_session()
        _CHAT_SESSIONS[key] = session

    session["updated_at"] = now

    # Đồng bộ ca bệnh từ context ảnh vào namespace DISEASE.
    ctx = getattr(body, "context", None)
    if ctx is not None:
        disease_context = getattr(ctx, "disease_context", None)
        disease_session = getattr(ctx, "disease_session", None)

        if disease_context or disease_session:
            session["disease"]["active"] = True

            disease_crop = _disease_crop_from_context(body)
            # Nếu disease context không chứa tên cây, context.crop lúc này là
            # fallback hợp lệ cho ca bệnh ảnh; KHÔNG dùng crop từ market.
            if not disease_crop:
                disease_crop = getattr(ctx, "crop", None)

            if disease_crop:
                session["disease"]["crop"] = disease_crop

            disease_name = _extract_disease_name(
                disease_context=disease_context,
                disease_session=disease_session,
            )
            if disease_name:
                session["disease"]["name"] = disease_name

    return session


def _runtime_explicit_topic(message: str) -> Optional[str]:
    """Topic được nói rõ trong CHÍNH câu hiện tại."""
    msg = message.lower().strip()

    if any(k in msg for k in [
        "vùng này có thể trồng cây gì", "vùng này trồng cây gì",
        "phù hợp trồng cây gì", "nên trồng cây gì", "cây gì phù hợp",
        "cây trồng phù hợp", "vùng này phù hợp"
    ]):
        return "suitability"

    market_terms = [
        "có nên bán", "nên bán", "bán luôn", "bán bây giờ",
        "bán hay giữ", "giữ lại", "xu hướng giá", "dự báo giá",
        "phân tích xu hướng", "chốt lời", "bán được không", "bán được ko",
        "bán dc không", "bán dc ko", "có bán được không", "có bán được ko",
        "nên chốt", "chốt bán", "đặt cọc", "thương lái",
        "có thể bán", "bán giờ", "bán lúc này", "bán hiện nay"
    ]
    if any(k in msg for k in market_terms):
        return "market"

    # Quy tắc tổng quát: có động từ "bán" = quyết định thị trường,
    # TRỪ khi người dùng hỏi rõ "giá bán".
    if "bán" in msg and "giá bán" not in msg:
        return "market"

    if re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:ngày|tuần|tháng)\s*(?:nữa)?\s*(?:thu hoạch|cắt|hái)\b",
        msg
    ):
        return "market"

    topic = _explicit_topic(message)
    if topic:
        return topic

    return None


def _is_generic_followup(message: str) -> bool:
    msg = message.lower().strip()
    if not msg or len(msg) > 100:
        return False

    if any(k in msg for k in [
        "hiện nay", "bây giờ", "lúc này", "thì sao", "còn",
        "vậy", "thế", "là sao", "rồi sao", "như thế nào",
        "thế nào", "sao", "ko", "không", "có"
    ]):
        return True

    if re.fullmatch(r"\d+(?:[.,]\d+)?\s*(?:ngày|tháng|năm|tuần)", msg):
        return True

    return len(msg.split()) <= 4


def _resolve_runtime_intent(message: str, body: ChatRequest, session: Dict[str, Any]) -> str:
    """
    Resolver chính:
    - explicit topic > pending slot > active topic > fallback classifier.
    - Không suy market crop sang disease crop.
    """
    explicit = _runtime_explicit_topic(message)
    current_crop = detect_crop_in_message(message)
    msg = message.lower().strip()

    # "bệnh cây như thế..." luôn quay về namespace disease của ảnh.
    if explicit == "disease":
        return "disease"

    # Nếu đang ở ca bệnh, hỏi tiếp tưới/bón/thuốc mà không nói cây mới
    # thì vẫn là treatment của ca bệnh.
    if explicit == "cultivation":
        treatment_words = [
            "tưới", "tưới tiêu", "bón", "phân", "phân bón",
            "thuốc", "dùng thuốc", "phun thuốc"
        ]
        if (
            session["active_topic"] == "disease"
            and session["disease"]["active"]
            and not current_crop
            and any(k in msg for k in treatment_words)
        ):
            return "disease"
        return "cultivation"

    if explicit:
        return explicit

    pending = session.get("pending", {})
    if (
        pending.get("field")
        and pending.get("topic")
        and _is_short_slot_answer(message, pending.get("field"))
    ):
        return pending["topic"]

    active = session.get("active_topic")

    if active == "market":
        if re.search(
            r"\b\d+(?:[.,]\d+)?\s*(?:ngày|tuần|tháng)\s*(?:nữa)?\b",
            msg
        ) and any(k in msg for k in ["thu hoạch", "cắt", "hái", "bán"]):
            return "market"

    # Bare crop kế thừa đúng namespace đang hoạt động.
    if current_crop and len(msg.split()) <= 4:
        if active in {"price", "market"}:
            return "price" if active == "price" else "market"
        if active == "disease":
            # Chỉ đổi crop bệnh nếu chatbot vừa hỏi "cây gì".
            if pending.get("field") == "crop":
                return "disease"
            # Nếu không hỏi crop, tên cây mới là chuyển sang canh tác cây đó.
            return "cultivation"
        if active == "cultivation":
            return "cultivation"

    if active and _is_generic_followup(message):
        return active

    # Fallback cũ chỉ dùng khi state không giải quyết được.
    return resolve_intent(message, body)


def _select_runtime_crop(
    message: str,
    intent: str,
    session: Dict[str, Any],
) -> Optional[str]:
    current_crop = detect_crop_in_message(message)
    pending = session.get("pending", {})

    if intent in {"price", "market"}:
        if current_crop:
            session["market"]["crop"] = current_crop
        return session["market"].get("crop")

    if intent == "disease":
        # Nếu đang trả lời câu "cây gì?" thì cho phép cập nhật disease crop.
        if current_crop and pending.get("field") == "crop":
            session["disease"]["crop"] = current_crop
        return session["disease"].get("crop") or current_crop

    if intent == "cultivation":
        if current_crop:
            session["cultivation"]["crop"] = current_crop
        return (
            session["cultivation"].get("crop")
            or current_crop
        )

    return current_crop


def _set_runtime_pending(session: Dict[str, Any], topic: str, crop: Optional[str], answer: str):
    field = _pending_from_assistant(answer)
    session["pending"] = {
        "field": field,
        "topic": topic if field else None,
        "crop": crop if field else None,
    }


def _finalize_runtime_session(
    session: Dict[str, Any],
    topic: str,
    crop: Optional[str],
    answer: str,
):
    session["active_topic"] = topic
    session["updated_at"] = time.time()
    session["last_answer"] = answer

    if topic in {"price", "market"} and crop:
        session["market"]["crop"] = crop

    elif topic == "disease":
        session["disease"]["active"] = True
        if crop:
            session["disease"]["crop"] = crop

    elif topic == "cultivation" and crop:
        session["cultivation"]["crop"] = crop

    _set_runtime_pending(session, topic, crop, answer)



def _capture_market_pending_answer(
    message: str,
    session: Dict[str, Any],
) -> Optional[str]:
    """
    Ghi nhận các slot market từ câu trả lời/ngữ cảnh mới của người dùng.
    """
    pending = session.get("pending", {})
    field = pending.get("field")
    msg = message.strip()
    msg_low = msg.lower()

    # Tuổi trái
    if field == "fruit_age":
        if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:ngày|tháng|tuần)\b", msg_low):
            session["market"]["fruit_age_text"] = msg
            session["pending"] = {
                "field": None,
                "topic": None,
                "crop": None,
            }
            return "fruit_age"

    # Thời gian còn lại đến thu hoạch:
    # "1 tháng nữa thu hoạch", "còn 20 ngày", "khoảng 3 tuần nữa cắt"
    harvest_match = re.search(
        r"(?:còn\s+|khoảng\s+)?(\d+(?:[.,]\d+)?)\s*(ngày|tuần|tháng)\s*(?:nữa)?"
        r"(?:\s*(?:là|thì))?\s*(?:thu hoạch|cắt|hái)?",
        msg_low,
    )
    if harvest_match and any(k in msg_low for k in ["thu hoạch", "cắt", "hái", "nữa", "bán"]):
        qty = harvest_match.group(1)
        unit = harvest_match.group(2)
        session["market"]["harvest_in_text"] = f"{qty} {unit}"
        return "harvest_in"

    return None


def _future_horizon_days(message: str) -> Optional[int]:
    m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(ngày|tuần|tháng)\s*(?:nữa)?\b", message.lower())
    if not m:
        return None
    n = float(m.group(1).replace(",", "."))
    return round(n if m.group(2) == "ngày" else n * 7 if m.group(2) == "tuần" else n * 30)


def _market_local_answer(crop: Optional[str], message: str, location: str, session: Dict[str, Any]) -> str:
    """
    Market dùng dữ liệu hệ thống trước, không phụ thuộc LLM cho câu bán/giữ cơ bản.
    """
    if not crop:
        return "Bạn đang muốn đánh giá nên bán loại nông sản nào?"

    # PHẢI bắt slot trước horizon.
    # Nếu chatbot vừa hỏi tuổi trái và user đáp "3 tháng" thì đó là TUỔI TRÁI,
    # không phải "giá 3 tháng nữa".
    slot = _capture_market_pending_answer(message, session)
    fruit_age_text = session.get("market", {}).get("fruit_age_text")
    harvest_in_text = session.get("market", {}).get("harvest_in_text")

    horizon_days = None if slot == "fruit_age" else _future_horizon_days(message)
    if horizon_days and horizon_days > 35 and not any(
        k in message.lower() for k in ["thu hoạch", "cắt", "hái"]
    ):
        current_price = get_latest_price(crop)
        price_part = (
            f"Giá hiện tại hệ thống đang ghi nhận khoảng **{current_price:,.0f} VNĐ/kg**. "
            if current_price is not None else ""
        )
        return (
            f"{price_part}Bạn đang hỏi mốc khoảng **{horizon_days} ngày nữa**. "
            "Mô hình hiện tại chỉ dự báo khoảng **30 ngày**, nên chưa đủ cơ sở để kết luận "
            "giá ở mốc này. Tôi không dùng dự báo 30 ngày để trả lời thay cho dự báo dài hơn."
        )

    current_price = get_latest_price(crop)
    if current_price is None:
        return (
            f"Hiện hệ thống chưa có dữ liệu giá đủ tin cậy cho **{crop}**, "
            "nên chưa thể đánh giá thời điểm bán một cách có căn cứ."
        )

    forecast_text = ""
    direction = None
    try:
        forecast_data = predict_price(crop, current_price, location)
        if forecast_data:
            pred_30d = float(forecast_data[-1]["predicted"])
            pct = ((pred_30d - current_price) / current_price) * 100
            if pct > 3:
                direction = "up"
                forecast_text = (
                    f" Mô hình nội bộ đang cho mức dự báo 30 ngày khoảng "
                    f"**{pred_30d:,.0f} VNĐ/kg**, cao hơn hiện tại khoảng **{pct:.1f}%**."
                )
            elif pct < -3:
                direction = "down"
                forecast_text = (
                    f" Mô hình nội bộ đang cho mức dự báo 30 ngày khoảng "
                    f"**{pred_30d:,.0f} VNĐ/kg**, thấp hơn hiện tại khoảng **{abs(pct):.1f}%**."
                )
            else:
                direction = "flat"
                forecast_text = (
                    f" Mô hình nội bộ đang cho mức dự báo 30 ngày khoảng "
                    f"**{pred_30d:,.0f} VNĐ/kg**, chưa chênh nhiều so với hiện tại."
                )
    except Exception as e:
        logger.warning(f"Market forecast unavailable: {e}")

    base = (
        f"Giá hệ thống đang ghi nhận cho **{crop}** là khoảng "
        f"**{current_price:,.0f} VNĐ/kg**.{forecast_text}"
    )

    if direction == "up":
        advice = (
            " Nếu hàng đã đạt chất lượng và bạn không cần tiền ngay, có thể cân nhắc "
            "bán một phần để chốt dòng tiền và giữ một phần để theo dõi thêm. "
            "Dự báo chỉ là mô hình hỗ trợ, không phải cam kết giá."
        )
    elif direction == "down":
        advice = (
            " Nếu hàng đã đạt tiêu chuẩn bán, phương án thận trọng là chốt một phần "
            "hoặc phần lớn lượng hàng thay vì giữ toàn bộ. "
            "Dự báo chỉ là mô hình hỗ trợ, không phải cam kết giá."
        )
    else:
        advice = (
            " Nếu hàng đã đạt tiêu chuẩn bán, có thể chia thành nhiều đợt bán để giảm "
            "rủi ro biến động giá thay vì đặt cược toàn bộ vào một thời điểm."
        )

    # Ri6: dùng cả tuổi trái và thời gian còn lại đến thu hoạch.
    if crop == "Sầu riêng Ri6":
        if harvest_in_text:
            asking_sell = any(k in message.lower() for k in [
                "bán", "chốt", "đặt cọc", "thương lái"
            ])
            if asking_sell:
                advice += (
                    f"\n\nBạn cho biết còn khoảng **{harvest_in_text} nữa mới thu hoạch**. "
                    "Nếu hỏi có nên bán/chốt trước ngay bây giờ thì tôi không khuyên khóa toàn bộ sản lượng quá sớm. "
                    "Với mức dự báo 30 ngày hiện chỉ nhỉnh hơn giá hiện tại vài phần trăm, phương án thận trọng là "
                    "chỉ chốt một phần nếu thương lái đưa giá và điều khoản tốt, còn lại theo dõi đến gần ngày thu hoạch. "
                    "Nếu chưa có thỏa thuận rõ về giá, chất lượng, tiền cọc và cách xử lý khi không đạt chuẩn thì chưa nên chốt toàn bộ."
                )
            else:
                advice += (
                    f"\n\nTôi đã ghi nhận: còn khoảng **{harvest_in_text} nữa mới thu hoạch**. "
                    "Thông tin này sẽ được giữ cho các câu hỏi tiếp theo về bán/chốt giá sầu riêng."
                )
        elif fruit_age_text:
            advice += (
                f"\n\nBạn vừa cho biết trái khoảng **{fruit_age_text}**. "
                "Tôi đã ghi nhận tuổi trái này cho mạch tư vấn hiện tại. "
                "Không nên quyết định cắt chỉ dựa vào giá hoặc số tháng; cần đối chiếu thêm "
                "độ già thực tế của trái và tiêu chuẩn thu hoạch của giống tại vườn. "
                "Nếu trái chưa đạt độ già thu hoạch thì tiếp tục chăm sóc; nếu đã đạt chuẩn, "
                "có thể cân nhắc bán từng phần theo mức giá và dự báo nêu trên."
            )
        else:
            advice += (
                "\n\nNếu đây là sầu riêng còn trên cây, cho tôi biết **tuổi trái tính từ khi đậu trái** "
                "(khoảng bao nhiêu ngày hoặc tháng) để đánh giá sát hơn."
            )

    return base + advice



def _is_disease_care_followup(message: str) -> bool:
    msg = message.lower().strip()
    return any(x in msg for x in [
        "tưới", "tưới tiêu", "nước", "bón phân", "phân bón",
        "bón", "dùng thuốc", "phun thuốc", "thuốc", "chăm sóc"
    ])


def _looks_like_generic_disease_fallback(answer: str) -> bool:
    if not answer:
        return True
    low = answer.lower()
    markers = [
        "cắt bỏ mô bệnh và vệ sinh dụng cụ cắt",
        "giảm độ ẩm trong tán",
        "không để vườn úng nước",
        "tăng cường sức khỏe cây bằng dinh dưỡng cân đối",
    ]
    return sum(x in low for x in markers) >= 3



def _suitability_fallback(location: str) -> str:
    return (
        f"Với **{location}**, hệ thống nên ưu tiên xem xét các nhóm cây đang có dữ liệu kỹ thuật "
        "phù hợp với điều kiện Lâm Đồng như **cà phê, chè và sầu riêng**. "
        "Ngoài ra có thể đánh giá thêm **bơ và măng cụt** nếu điều kiện đất, độ cao và thoát nước phù hợp.\n\n"
        "Để chọn chính xác cây nào cho một vị trí cụ thể, cần thêm ít nhất: "
        "**độ cao, loại đất/độ dốc, khả năng thoát nước và diện tích vườn**. "
        "Tôi không nên lấy ngữ cảnh giá cà phê trước đó để trả lời câu hỏi vùng trồng."
    )


def _llm_unavailable_answer(answer: str) -> bool:
    if not answer:
        return True
    low = answer.lower()
    return any(k in low for k in [
        "dịch vụ tạo câu trả lời đang tạm gián đoạn",
        "hệ thống đang bảo trì phần tư vấn",
        "ai đang bận",
        "chưa tìm thấy tài liệu kỹ thuật đủ phù hợp",
    ])


def _disease_care_fallback(crop: Optional[str], disease_name: Optional[str]) -> str:
    crop = crop or "cây trong ca bệnh hiện tại"
    disease_name = disease_name or "bệnh đã nhận diện"
    return (
        f"Với **{crop}** đang bị **{disease_name}**, về tưới tiêu và phân bón:\n\n"
        "**Tưới tiêu:** Nếu đang mưa hoặc đất còn ẩm thì không tưới thêm. "
        "Khơi thông rãnh, không để nước đọng quanh gốc. Khi đất bắt đầu khô mới tưới vừa đủ "
        "vùng rễ và tránh làm ướt tán lá.\n\n"
        "**Phân bón:** Không bón dồn hoặc bón thừa đạm khi cây đang bị bệnh. "
        "Duy trì dinh dưỡng cân đối theo giai đoạn sinh trưởng và chỉ bón khi đất thoát nước tốt. "
        "Nếu cây đang mang trái gần thu hoạch, giữ nước ổn định, tránh lúc quá khô lúc quá ướt "
        "và không tự tăng liều phân để thúc trái."
    )




DISEASE_DISPLAY_NAMES = {
    "canker_disease": "bệnh loét thân/cành",
    "anthracnose_disease": "bệnh thán thư",
    "fruit_rot": "bệnh thối trái",
    "pink_disease": "bệnh nấm hồng",
    "sooty_mold": "bệnh muội đen",
    "stem_blight": "bệnh cháy thân/cành",
    "stem_cracking_ gummosis": "bệnh nứt thân, chảy gôm",
    "thrips_disease": "bọ trĩ gây hại",
    "yellow_leaf": "bệnh vàng lá",
    "mealybug_infestation": "rệp sáp gây hại",
    "Leaf_Phomopsis": "bệnh lá do Phomopsis",
    "Leaf_Algal": "bệnh đốm rong",
    "Leaf_Blight": "bệnh cháy lá",
    "Leaf_Colletotrichum": "bệnh lá do Colletotrichum",
    "Leaf_Rhizoctonia": "bệnh lá do Rhizoctonia",
    "Leaf_Healthy": "lá khỏe",
}

def _display_disease_name(name: Optional[str]) -> str:
    if not name:
        return "bệnh đã nhận diện"
    return DISEASE_DISPLAY_NAMES.get(name, name.replace("_", " "))

def _disease_followup_kinds(message: str) -> List[str]:
    """
    Một câu có thể hỏi nhiều ý cùng lúc:
    'tưới tiêu phân thuốc thế nào' -> irrigation + fertilizer + medicine.
    """
    msg = message.lower().strip()
    kinds: List[str] = []

    if any(k in msg for k in [
        "tên thuốc", "thuốc gì", "thuốc nào", "loại thuốc",
        "hoạt chất gì", "hoạt chất nào"
    ]):
        kinds.append("medicine")

    if any(k in msg for k in [
        "liều", "liều lượng", "pha bao nhiêu", "nồng độ",
        "bao nhiêu ml", "bao nhiêu gam"
    ]):
        kinds.append("dose")

    if any(k in msg for k in [
        "phun thế nào", "phun sao", "cách phun", "bao lâu phun",
        "mấy ngày phun", "phun lại", "thời điểm phun"
    ]):
        kinds.append("spray")

    if any(k in msg for k in [
        "phân gì", "bón gì", "bón phân", "phân bón", "phân"
    ]):
        kinds.append("fertilizer")

    if any(k in msg for k in [
        "tưới", "tưới tiêu", "nước"
    ]):
        kinds.append("irrigation")

    # Từ "thuốc" đơn lẻ cũng là medicine nếu chưa bắt được medicine/spray/dose.
    if "thuốc" in msg and not any(k in kinds for k in ["medicine", "dose", "spray"]):
        kinds.append("medicine")

    # Bỏ trùng, giữ thứ tự.
    return list(dict.fromkeys(kinds))


def _disease_followup_kind(message: str) -> str:
    """Tương thích code cũ: trả subtopic đầu tiên hoặc general."""
    kinds = _disease_followup_kinds(message)
    return kinds[0] if kinds else "general"


def _disease_subtopic_answer(crop: Optional[str], disease_name: Optional[str], kind: str) -> str:
    crop_text = crop or "cây trong ca bệnh hiện tại"
    disease_text = _display_disease_name(disease_name)
    actives = {
        "canker_disease": ["gốc đồng (Copper hydroxide/Copper oxychloride)", "Mancozeb"],
        "anthracnose_disease": ["Azoxystrobin", "Difenoconazole", "Mancozeb"],
        "fruit_rot": ["gốc đồng", "Mancozeb"],
        "pink_disease": ["gốc đồng", "Difenoconazole"],
        "stem_blight": ["gốc đồng", "Mancozeb"],
        "stem_cracking_ gummosis": ["gốc đồng"],
        "Leaf_Phomopsis": ["Mancozeb", "Copper hydroxide", "Azoxystrobin", "Difenoconazole"],
        "Leaf_Blight": ["Mancozeb", "gốc đồng"],
        "Leaf_Colletotrichum": ["Azoxystrobin", "Difenoconazole", "Mancozeb"],
        "Leaf_Rhizoctonia": ["Azoxystrobin"],
    }.get(disease_name or "", [])

    if kind == "medicine":
        if not actives:
            return f"Với **{crop_text}** đang bị **{disease_text}**, dữ liệu hiện tại chưa đủ để nêu hoạt chất cụ thể mà không đoán. Hãy chọn thuốc BVTV có đăng ký đúng cây và đối tượng gây hại trên nhãn."
        return f"Với **{crop_text}** đang bị **{disease_text}**, có thể tham khảo các **hoạt chất**: **{', '.join(actives)}**. Khi mua cần chọn sản phẩm được đăng ký cho đúng cây/đối tượng và dùng đúng hướng dẫn trên nhãn."
    if kind == "dose":
        return f"Với **{crop_text}** đang bị **{disease_text}**, liều phụ thuộc sản phẩm và hàm lượng hoạt chất. Không nên tự đặt một liều chung. Hãy dùng đúng liều trên nhãn; nếu bạn cho tên sản phẩm và hàm lượng, hệ thống mới nên đối chiếu liều tương ứng."
    if kind == "spray":
        return f"Với **{crop_text}** đang bị **{disease_text}**, phun khi lá khô và trời tạnh, phủ đều vùng bệnh và vùng lân cận theo nhãn. Không tự tăng nồng độ hoặc pha nhiều thuốc khi chưa kiểm tra khả năng phối trộn."
    if kind == "fertilizer":
        return f"Với **{crop_text}** đang bị **{disease_text}**, tránh bón dồn hoặc thừa đạm; duy trì dinh dưỡng cân đối theo giai đoạn cây và chỉ bón khi vùng rễ thoát nước tốt."
    if kind == "irrigation":
        return f"Với **{crop_text}** đang bị **{disease_text}**, nếu đất còn ẩm hoặc đang mưa thì không tưới thêm; ưu tiên thoát nước. Khi đất khô mới tưới vừa đủ vùng rễ và tránh làm ướt tán kéo dài."
    return ""


def _disease_multi_subtopic_answer(
    crop: Optional[str],
    disease_name: Optional[str],
    message: str,
) -> str:
    kinds = _disease_followup_kinds(message)
    if not kinds:
        return ""

    parts = []
    labels = {
        "irrigation": "Tưới tiêu",
        "fertilizer": "Phân bón",
        "medicine": "Thuốc/hoạt chất",
        "dose": "Liều lượng",
        "spray": "Cách phun",
    }

    for kind in kinds:
        ans = _disease_subtopic_answer(crop, disease_name, kind)
        if ans:
            # Với câu đa ý, bỏ phần mở đầu lặp lại bằng cách giữ toàn câu nhưng gắn nhãn.
            parts.append(f"**{labels.get(kind, kind)}:** {ans}")

    return "\\n\\n".join(parts)




def _disease_semantic_followup_kind(message: str) -> str:
    """Nhận diện câu hỏi nguyên nhân hoặc ảnh hưởng thời tiết của ca bệnh hiện tại."""
    msg = (message or "").lower().strip()

    cause_terms = [
        "nguyên nhân", "do đâu", "do gì", "tại sao bị", "vì sao bị",
        "vì sao cây bị", "bệnh này do gì", "bệnh này do đâu",
    ]
    if any(term in msg for term in cause_terms):
        return "cause"

    weather_risk_terms = [
        "có nặng thêm không", "có nặng hơn không", "có phát triển mạnh không",
        "có lan nhanh không", "thời tiết này có ảnh hưởng", "thời tiết hiện nay",
        "với thời tiết hiện nay", "mưa như vậy có ảnh hưởng",
        "độ ẩm như vậy có ảnh hưởng", "mưa nhiều có làm bệnh",
        "ẩm như vậy có làm bệnh",
    ]
    if any(term in msg for term in weather_risk_terms):
        return "weather_risk"

    return ""


def _disease_cause_answer(crop: Optional[str], disease_name: Optional[str]) -> str:
    crop_text = crop or "cây trong ca bệnh hiện tại"
    disease_text = _display_disease_name(disease_name)
    raw = (disease_name or "").lower()

    if "canker" in raw or "loét" in disease_text.lower():
        return (
            f"Với **{crop_text}** đang được nhận diện là **{disease_text}**, "
            "bệnh thường phát sinh khi tác nhân gây bệnh xâm nhập qua mô thân/cành, "
            "đặc biệt tại các vết thương cơ giới, vị trí nứt hoặc mô cây suy yếu. "
            "Mưa nhiều, độ ẩm cao, tán cây ẩm kéo dài và thoát nước kém có thể làm bệnh "
            "dễ phát triển và lây lan hơn. Ảnh nhận diện chỉ là bước hỗ trợ; để xác định "
            "chính xác tác nhân cần kết hợp biểu hiện thực tế trên thân/cành và điều kiện vườn."
        )

    return (
        f"Với **{crop_text}** đang được nhận diện là **{disease_text}**, nguyên nhân có thể "
        "liên quan đến tác nhân nấm, vi khuẩn, côn trùng hoặc điều kiện canh tác bất lợi tùy bệnh. "
        "Cần đối chiếu thêm vị trí vết bệnh, tốc độ lan, độ ẩm vườn và tình trạng thân/rễ để xác định sát hơn."
    )


def _disease_weather_risk_answer(
    crop: Optional[str],
    disease_name: Optional[str],
    location: str,
    weather: Any,
) -> str:
    crop_text = crop or "cây trong ca bệnh hiện tại"
    disease_text = _display_disease_name(disease_name)

    if not isinstance(weather, dict):
        return (
            f"Hiện hệ thống chưa lấy được dữ liệu thời tiết đủ tin cậy tại **{location}** "
            f"để đánh giá nguy cơ của **{disease_text}** trên **{crop_text}**."
        )

    rain = _weather_number(weather, "precipitation")
    humidity = _weather_number(weather, "humidity")
    tmin = _weather_number(weather, "temp_min")
    tmax = _weather_number(weather, "temp_max")

    details = []
    if tmin is not None and tmax is not None:
        details.append(f"nhiệt độ khoảng **{tmin:g}–{tmax:g}°C**")
    if rain is not None:
        details.append(f"lượng mưa khoảng **{rain:g} mm**")
    if humidity is not None:
        details.append(f"độ ẩm khoảng **{humidity:g}%**")
    weather_text = ", ".join(details) if details else "chưa có đủ chỉ số chi tiết"

    high_moisture = (
        (rain is not None and rain >= 10)
        or (humidity is not None and humidity >= 80)
    )

    if high_moisture:
        return (
            f"**Có nguy cơ bệnh nặng thêm.** Với **{crop_text}** đang bị **{disease_text}**, "
            f"thời tiết hiện tại tại **{location}** ({weather_text}) đang khá ẩm. "
            "Điều kiện ẩm kéo dài có thể giúp mô bệnh duy trì ẩm và làm tăng nguy cơ bệnh tiếp tục "
            "phát triển hoặc lây lan. Nên ưu tiên thoát nước, giữ tán thông thoáng, hạn chế làm ướt "
            "thân/cành và theo dõi vết bệnh sau các đợt mưa."
        )

    return (
        f"Với **{crop_text}** đang bị **{disease_text}**, thời tiết hiện tại tại **{location}** "
        f"({weather_text}) chưa cho thấy áp lực ẩm quá cao. Tuy vậy vẫn cần theo dõi vết bệnh "
        "và tránh để thân/cành hoặc vùng rễ ẩm kéo dài."
    )


def _weather_followup_kind(message: str) -> str:
    msg = message.lower().strip()

    irrigation_phrases = [
        "có nên tưới", "nên tưới", "tưới không", "tưới ko",
        "tưới thế nào", "tưới sao", "cần tưới", "có cần tưới"
    ]
    if any(k in msg for k in irrigation_phrases):
        return "irrigation"

    rain_effect_phrases = [
        "mưa gần đây ảnh hưởng", "mưa ảnh hưởng", "mưa tác động",
        "mưa nhiều ảnh hưởng", "mưa thế này ảnh hưởng",
        "ảnh hưởng cây thế nào", "ảnh hưởng đến cây thế nào"
    ]
    if any(k in msg for k in rain_effect_phrases):
        return "rain_effect"

    return "current_weather"


def _weather_number(weather: Any, key: str):
    if not isinstance(weather, dict):
        return None
    value = weather.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _weather_irrigation_answer(location: str, weather: Any) -> str:
    rain = _weather_number(weather, "precipitation")
    humidity = _weather_number(weather, "humidity")

    if rain is None and humidity is None:
        return (
            f"Hiện hệ thống chưa có đủ dữ liệu mưa và độ ẩm tại **{location}** "
            "để kết luận có nên tưới hay không. Hãy kiểm tra độ ẩm đất vùng rễ trước khi tưới."
        )

    # Quy tắc bảo thủ: mưa đáng kể hoặc độ ẩm không khí rất cao -> chưa tưới bổ sung.
    if (rain is not None and rain >= 10) or (humidity is not None and humidity >= 80):
        evidence = []
        if rain is not None:
            evidence.append(f"lượng mưa khoảng **{rain:g} mm**")
        if humidity is not None:
            evidence.append(f"độ ẩm khoảng **{humidity:g}%**")
        return (
            f"Với thời tiết hiện tại tại **{location}**, **chưa nên tưới thêm**. "
            + " và ".join(evidence)
            + " cho thấy điều kiện đang khá ẩm. Ưu tiên kiểm tra và khơi thông thoát nước; "
              "chỉ tưới lại khi đất vùng rễ bắt đầu khô, tránh để cây bị úng."
        )

    if rain is not None and rain < 2 and (humidity is None or humidity < 70):
        return (
            f"Tại **{location}**, lượng mưa hiện chỉ khoảng **{rain:g} mm**"
            + (f" và độ ẩm khoảng **{humidity:g}%**" if humidity is not None else "")
            + ". Có thể cần tưới nếu kiểm tra thấy đất vùng rễ đã khô; tưới vừa đủ, "
              "không tưới theo lịch cứng khi đất vẫn còn ẩm."
        )

    return (
        f"Điều kiện tại **{location}** chưa đủ để quyết định chỉ bằng số liệu thời tiết. "
        "Hãy kiểm tra đất vùng rễ: còn ẩm thì chưa tưới, bắt đầu khô mới tưới vừa đủ; "
        "đồng thời bảo đảm thoát nước tốt."
    )


def _weather_rain_effect_answer(location: str, weather: Any) -> str:
    rain = _weather_number(weather, "precipitation")
    humidity = _weather_number(weather, "humidity")

    evidence = []
    if rain is not None:
        evidence.append(f"lượng mưa khoảng **{rain:g} mm**")
    if humidity is not None:
        evidence.append(f"độ ẩm khoảng **{humidity:g}%**")

    if (rain is not None and rain >= 10) or (humidity is not None and humidity >= 80):
        prefix = (
            f"Tại **{location}**, " + " và ".join(evidence) + ". "
            if evidence else f"Tại **{location}**, "
        )
        return (
            prefix
            + "Điều kiện ẩm kéo dài có thể làm đất khó thoát nước, tăng nguy cơ úng rễ "
              "và tạo môi trường thuận lợi cho một số bệnh nấm phát triển. "
              "Nên kiểm tra rãnh thoát nước, hạn chế tưới bổ sung, giữ tán thông thoáng "
              "và theo dõi sớm các dấu hiệu bệnh trên lá, cành và thân."
        )

    if evidence:
        return (
            f"Tại **{location}**, " + " và ".join(evidence) + ". "
            "Mức ảnh hưởng còn phụ thuộc loại cây và khả năng thoát nước của vườn. "
            "Nên kiểm tra độ ẩm đất và tình trạng lá/rễ trước khi điều chỉnh tưới."
        )

    return (
        f"Hiện hệ thống chưa có đủ số liệu mưa tại **{location}** để đánh giá tác động. "
        "Nếu bạn cho biết loại cây, hệ thống có thể tư vấn sát hơn theo cây trồng."
    )


def _weather_deterministic_answer(message: str, location: str, weather: Any) -> str:
    kind = _weather_followup_kind(message)
    if kind == "irrigation":
        return _weather_irrigation_answer(location, weather)
    if kind == "rain_effect":
        return _weather_rain_effect_answer(location, weather)
    return _weather_local_answer(location, weather)


def _weather_local_answer(location: str, weather: Any) -> str:
    """Trả thời tiết trực tiếp từ service, không phụ thuộc Gemini/LM Studio."""
    if not isinstance(weather, dict):
        return f"Hiện chưa đọc được dữ liệu thời tiết của **{location}**."

    tmin = weather.get("temp_min")
    tmax = weather.get("temp_max")
    rain = weather.get("precipitation")
    humidity = weather.get("humidity")

    details = []
    if tmin is not None and tmax is not None:
        details.append(f"nhiệt độ khoảng **{tmin}–{tmax}°C**")
    elif tmax is not None:
        details.append(f"nhiệt độ cao nhất khoảng **{tmax}°C**")

    if rain is not None:
        details.append(f"lượng mưa khoảng **{rain} mm**")
    if humidity is not None:
        details.append(f"độ ẩm khoảng **{humidity}%**")

    if not details:
        return f"Hệ thống đã kết nối dữ liệu thời tiết **{location}** nhưng chưa có đủ chỉ số để hiển thị."

    return f"Thời tiết hệ thống đang ghi nhận tại **{location}**: " + ", ".join(details) + "."


def _disease_contextual_fallback(
    crop: Optional[str],
    disease_name: Optional[str],
    message: str,
) -> str:
    crop = crop or "cây trong ca bệnh hiện tại"
    raw_disease_name = disease_name
    disease_name = _display_disease_name(disease_name)
    msg = message.lower()

    kind = _disease_followup_kind(message)
    if kind != "general":
        return _disease_subtopic_answer(crop, raw_disease_name, kind)

    if any(k in msg for k in ["tưới", "bón", "phân", "thuốc", "phun"]):
        return (
            f"Với **{crop}** đang bị **{disease_name}**:\n\n"
            "**Tưới tiêu:** nếu đất còn ẩm hoặc đang có mưa thì không tưới thêm; "
            "ưu tiên thoát nước và tránh làm ướt tán lá. Khi đất khô mới tưới vừa đủ vùng rễ.\n\n"
            "**Phân bón:** không bón dồn hoặc bón thừa đạm lúc cây đang bệnh. "
            "Duy trì dinh dưỡng cân đối theo giai đoạn của cây, chỉ bón khi đất thoát nước tốt.\n\n"
            "**Thuốc BVTV:** chỉ dùng sản phẩm có đăng ký cho cây/bệnh tương ứng và làm đúng nhãn. "
            "Với bệnh nấm, ưu tiên xử lý vệ sinh vườn, cắt bỏ mô bệnh và dùng thuốc nấm phù hợp theo hướng dẫn; "
            "không tự tăng liều hoặc pha nhiều thuốc khi chưa kiểm tra khả năng phối trộn."
        )

    return (
        f"Đây vẫn là ca **{crop}** đã nhận diện **{disease_name}**. "
        "Bạn nên cắt bỏ lá/cành bị bệnh nặng, vệ sinh và khử trùng dụng cụ, "
        "tỉa tán cho thông thoáng, khơi rãnh thoát nước và tránh để tán lá ẩm kéo dài. "
        "Hạn chế bón thừa đạm khi bệnh đang phát triển. Nếu bệnh tiếp tục lan, "
        "nên dùng thuốc nấm có đăng ký cho cây/bệnh tương ứng theo đúng nhãn và thời gian cách ly."
    )


def _answer_has_disease_context(
    answer: str,
    crop: Optional[str],
    disease_name: Optional[str],
) -> bool:
    low = (answer or "").lower()
    crop_ok = True
    disease_ok = True

    if crop:
        crop_tokens = [x for x in crop.lower().split() if len(x) >= 3]
        crop_ok = any(tok in low for tok in crop_tokens)

    if disease_name:
        disease_tokens = [
            x for x in disease_name.lower().replace("bệnh", "").split()
            if len(x) >= 4
        ]
        disease_ok = any(tok in low for tok in disease_tokens)

    return crop_ok and disease_ok


async def _build_chat_answer(message: str, body: ChatRequest, user_id=None):
    """Bộ điều phối theo session state thật, không dựng lại topic từ đầu mỗi câu."""
    session = _get_runtime_session(user_id, body)
    intent = _resolve_runtime_intent(message, body, session)
    crop = _select_runtime_crop(message, intent, session)

    def finish(answer: str, suggestions, topic: str, crop_value=None):
        _finalize_runtime_session(
            session,
            topic=topic,
            crop=crop_value,
            answer=answer,
        )
        return answer, suggestions, topic

    if intent == "greeting":
        session["active_topic"] = None
        session["pending"] = {"field": None, "topic": None, "crop": None}
        return (
            "Xin chào! Hôm nay tôi có thể giúp gì cho bạn?",
            ["Giá nông sản hôm nay?", "Tư vấn chăm sóc cây", "Nhận diện sâu bệnh"],
            "greeting",
        )

    if intent == "thanks":
        return finish(
            "Rất vui vì đã hỗ trợ được bạn. Chúc bạn chăm sóc vườn hiệu quả và có một ngày tốt lành! 🌱",
            [],
            "thanks",
            crop,
        )

    if intent == "price_list":
        answer, suggestions = _price_list_answer()
        return finish(answer, suggestions, "price", session["market"].get("crop"))

    if intent == "price":
        # Dùng crop của namespace market nếu câu hiện tại là follow-up ngắn.
        if crop and not detect_crop_in_message(message):
            current_price = get_latest_price(crop)
            if current_price is None:
                answer = (
                    f"Hiện hệ thống chưa có dữ liệu giá đủ tin cậy cho **{crop}** "
                    "để cung cấp giá hôm nay."
                )
                suggestions = []
            else:
                answer = (
                    f"Giá hiện tại hệ thống đang ghi nhận cho **{crop}** "
                    f"là khoảng **{current_price:,.0f} VNĐ/kg**."
                )
                suggestions = [
                    "Giá 30 ngày tới thế nào?",
                    "Có nên bán lúc này không?",
                    "Phân tích xu hướng giá",
                ]
        else:
            answer, suggestions = _price_answer(message, body)

        # Nếu _price_answer vừa nhận crop từ message, đồng bộ lại.
        detected = detect_crop_in_message(message)
        if detected:
            crop = detected
            session["market"]["crop"] = detected

        return finish(answer, suggestions, "price", crop)

    location = (
        body.context.location
        if body.context and body.context.location
        else "Phường B'Lao"
    )

    if intent == "market":
        crop = crop or session["market"].get("crop")
        answer = _market_local_answer(crop, message, location, session)
        session["market"]["action"] = (
            "sell" if "bán" in message.lower() or session["market"].get("action") == "sell"
            else "trend"
        )
        return finish(
            answer,
            ["Giá hiện tại?", "Phân tích xu hướng giá", "Có nên bán lúc này?"],
            "market",
            crop,
        )

    if intent == "suitability":
        weather_context = get_weather(location)
        answer = await _get_rag_response(
            f"Khu vực {location} phù hợp trồng cây gì? Phân tích theo khí hậu, đất đai, "
            "độ cao, lượng mưa, khả năng thoát nước và tài liệu kỹ thuật trong hệ thống.",
            history=[],
            disease_context=None,
            disease_session=None,
            weather_context=weather_context,
        )
        if _llm_unavailable_answer(answer):
            answer = _suitability_fallback(location)
        return finish(answer, [], "suitability", None)

    # Nếu đang có ca bệnh hoạt động, ưu tiên hiểu các câu hỏi về nguyên nhân
    # hoặc tác động của thời tiết lên bệnh trước nhánh weather chung.
    disease_semantic_kind = _disease_semantic_followup_kind(message)
    if session["disease"].get("active") and disease_semantic_kind == "cause":
        disease_crop = session["disease"].get("crop") or crop
        disease_name = session["disease"].get("name")
        answer = _disease_cause_answer(disease_crop, disease_name)
        return finish(answer, [], "disease", disease_crop)

    if session["disease"].get("active") and disease_semantic_kind == "weather_risk":
        disease_crop = session["disease"].get("crop") or crop
        disease_name = session["disease"].get("name")
        weather_context = get_weather(location)
        answer = _disease_weather_risk_answer(
            disease_crop,
            disease_name,
            location,
            weather_context,
        )
        return finish(answer, [], "disease", disease_crop)

    if intent == "weather":
        weather_context = get_weather(location)
        answer = _weather_deterministic_answer(message, location, weather_context)
        return finish(
            answer,
            [
                "Thời tiết này có nên tưới không?",
                "Mưa gần đây ảnh hưởng cây thế nào?",
                "Vùng này phù hợp trồng cây gì?",
            ],
            "weather",
            crop,
        )

    if intent == "finance":
        crop = (
            crop
            or session["market"].get("crop")
            or (body.context.crop if body.context else "Sầu riêng Ri6")
        )

        if get_latest_price(crop) is None:
            return finish(
                f"Hiện hệ thống chưa có dữ liệu giá đủ tin cậy cho **{crop}**, "
                "nên chưa thể tính mô phỏng tài chính chính xác cho cây này.",
                [],
                "finance",
                crop,
            )

        cap = body.context.capital if body.context and body.context.capital else 200_000_000
        area = body.context.area_ha if body.context and body.context.area_ha else 1.0
        mode = body.context.mode if body.context and body.context.mode else "Kinh doanh"

        answer = _run_finance_simulation(crop, cap, area, location, mode)
        return finish(
            answer,
            ["Rủi ro thời tiết?", "Giá nông sản hiện tại?", "Kỹ thuật chăm sóc?"],
            "finance",
            crop,
        )

    # Disease namespace: crop/name lấy từ ca bệnh ảnh, không lấy market crop.
    use_disease_context = intent == "disease"
    if use_disease_context:
        crop = session["disease"].get("crop") or crop

        # Drill-down của cùng ca bệnh: chỉ trả đúng ý đang hỏi,
        # không lặp lại toàn bộ tưới/phân/thuốc.
        disease_kinds = _disease_followup_kinds(message)
        if session["disease"].get("active") and disease_kinds:
            answer = _disease_multi_subtopic_answer(
                crop,
                session["disease"].get("name"),
                message,
            )
            return finish(answer, [], "disease", crop)

    weather_context = get_weather(location)

    state_for_prompt = {
        "crop": crop,
        "pending": session.get("pending", {}).get("field"),
    }

    if intent == "disease":
        disease_name = session["disease"].get("name")
        prefix = []
        if crop:
            prefix.append(f"Cây của ca bệnh: {crop}")
        if disease_name:
            prefix.append(f"Bệnh đã nhận diện trong ca ảnh: {disease_name}")
        prefix.append(f"Câu hỏi hiện tại: {message}")
        rag_query = ". ".join(prefix)
    else:
        rag_query = _enrich_followup_query(message, state_for_prompt, intent)

    answer = await _get_rag_response(
        rag_query,
        history=_history_for_topic(body.history, intent),
        disease_context=(
            body.context.disease_context
            if use_disease_context and body.context else None
        ),
        disease_session=(
            body.context.disease_session
            if use_disease_context and body.context else None
        ),
        weather_context=weather_context,
    )

    if intent == "disease" and session["disease"].get("active"):
        disease_crop = session["disease"].get("crop") or crop
        disease_name = session["disease"].get("name")

        # Không chấp nhận câu fallback chung chung hoặc câu không còn nhắc đúng ca bệnh.
        if (
            _llm_unavailable_answer(answer)
            or _looks_like_generic_disease_fallback(answer)
            or not _answer_has_disease_context(answer, disease_crop, disease_name)
        ):
            answer = _disease_contextual_fallback(
                disease_crop,
                disease_name,
                message,
            )

    return finish(answer, [], intent, crop)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    try:
        message = body.message.strip()
        if not message:
            return ChatResponse(
                answer="Xin chào! Hôm nay tôi có thể giúp gì cho bạn?",
                suggestions=["Giá nông sản hôm nay?", "Tư vấn chăm sóc cây", "Nhận diện sâu bệnh"],
            )

        msg_low = message.lower()
        forbidden = ["chính trị", "tôn giáo", "mã nguồn", "password"]
        if any(f in msg_low for f in forbidden):
            return ChatResponse(
                answer="Hệ thống chỉ hỗ trợ các câu hỏi liên quan đến nông nghiệp. Cảm ơn bạn.",
                suggestions=["Kỹ thuật trồng sầu riêng", "Giá cà phê hôm nay"],
            )

        answer, suggestions, _ = await _build_chat_answer(message, body, current_user.id)

        new_chat = models.ChatHistory(
            user_id=current_user.id,
            question=message,
            answer=answer,
        )
        db.add(new_chat)
        db.commit()

        return ChatResponse(answer=answer, suggestions=suggestions)

    except Exception as e:
        logger.error(f"Chat Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi hệ thống")


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    try:
        message = body.message.strip()

        if not message:
            return StreamingResponse(
                iter(["Xin chào! Hôm nay tôi có thể giúp gì cho bạn?"]),
                media_type="text/plain",
            )

        msg_low = message.lower()
        forbidden = ["chính trị", "tôn giáo", "mã nguồn", "password"]
        if any(f in msg_low for f in forbidden):
            return StreamingResponse(
                iter(["Hệ thống chỉ hỗ trợ các câu hỏi liên quan đến nông nghiệp. Cảm ơn bạn."]),
                media_type="text/plain",
            )

        answer, _, _ = await _build_chat_answer(message, body, current_user.id)

        try:
            new_chat = models.ChatHistory(
                user_id=current_user.id,
                question=message,
                answer=answer,
            )
            db.add(new_chat)
            db.commit()
        except Exception as db_error:
            db.rollback()
            logger.warning(f"Không lưu được lịch sử chat stream: {db_error}")

        return StreamingResponse(iter([answer]), media_type="text/plain")

    except Exception as e:
        logger.error(f"Stream Error: {e}", exc_info=True)
        return StreamingResponse(iter(["Lỗi kết nối server."]), media_type="text/plain")



@router.post("/chat/session/reset")
async def reset_chat_session(
    current_user: models.User = Depends(get_current_user),
):
    """
    Frontend gọi endpoint này khi người dùng đóng hộp chat.
    Khi đó topic/pending của phiên hiện tại được xóa sạch.
    """
    _CHAT_SESSIONS.pop(_session_key(current_user.id), None)
    return {"ok": True}


@router.get("/chat/history")
async def get_chat_history(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    history = db.query(models.ChatHistory).filter(
        models.ChatHistory.user_id == current_user.id
    ).order_by(models.ChatHistory.created_at.asc()).limit(50).all()
    return history