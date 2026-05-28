import os
import re
import logging
from typing import List, Optional
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
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
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
    msg = message.lower().strip()
    agri_keywords = ["cây", "trồng", "bón", "phân", "sâu", "bệnh", "tưới", "đất", "giống", "chăm sóc", "kỹ thuật", "cà phê", "sầu riêng", "chè", "trà"]
    
    if any(kw in msg for kw in _FINANCE_KEYWORDS):
        return "finance"
    if any(kw in msg for kw in agri_keywords) or len(msg) > 30:
        return "agriculture"

    if any(re.search(rf"\b{re.escape(g)}\b", msg) for g in _GREETING_WORDS): 
        return "greeting"
    if any(re.search(rf"\b{re.escape(t)}\b", msg) for t in _THANKS_WORDS): 
        return "thanks"
        
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
        "trà": "Chè Ô Long"
    }
    for kw, full_name in crop_keywords.items():
        if kw in msg:
            return full_name
    return None

async def _get_rag_response(query: str) -> str:
    # Filter nội dung nhạy cảm
    query_lower = query.lower()
    forbidden = ["quên hết", "chính trị", "tôn giáo", "mã nguồn", "password"]
    if any(k in query_lower for k in forbidden):
        return "Xin lỗi, tôi chỉ hỗ trợ các vấn đề về kỹ thuật canh tác nông nghiệp tại Lâm Đồng."

    # Tăng top_k lên 5 để lấy ngữ cảnh đầy đủ hơn
    context = rag_engine.get_relevant_context(query, top_k=5)
    
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
                "PHONG CÁCH: Thân thiện, am hiểu địa phương, sử dụng ngôn từ gần gũi với nhà nông (như gọi 'Bà con', chúc 'Mùa màng bội thu').\n\n"
                "NHIỆM VỤ: Giải đáp các thắc mắc về kỹ thuật canh tác, phòng trừ sâu bệnh cho 3 cây trồng chủ lực: Cà phê, Sầu riêng, và Chè Ô Long.\n\n"
                "NGỮ CẢNH TỪ SỔ TAY KỸ THUẬT:\n{context}\n\n"
                "QUY TẮC PHẢN HỒI:\n"
                "1. Ưu tiên tuyệt đối thông tin từ NGỮ CẢNH cung cấp phía trên.\n"
                "2. Nếu thông tin không có trong sổ tay, hãy dựa trên kiến thức nông nghiệp chuẩn xác nhưng phải mở đầu bằng: 'Dựa trên kinh nghiệm canh tác chung...'.\n"
                "3. Trình bày câu trả lời theo cấu trúc:\n"
                "   - Nhận định vấn đề (AI hiểu bà con đang hỏi gì).\n"
                "   - Hướng dẫn kỹ thuật chi tiết (Chia theo các bước hoặc gạch đầu dòng).\n"
                "   - Lưu ý quan trọng (Về thời tiết, liều lượng phân bón/thuốc hoặc an toàn).\n"
                "4. Luôn kết thúc bằng một lời chúc tốt đẹp cho vụ mùa.\n"
                "5. Tuyệt đối KHÔNG trả lời các vấn đề chính trị, tôn giáo hoặc nội dung không liên quan đến nông nghiệp."
            )),
            ("human", "{query}")
        ])
        chain = prompt | gemini_llm | StrOutputParser()
        try:
            res = await chain.ainvoke({"query": query, "context": context_str})
            if res:
                res = re.sub(r'<thought>.*?</thought>', '', res, flags=re.DOTALL)
                return res.strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg or "resourceexhausted" in error_msg:
                logger.warning(f"Gemini Rate Limit hit: {e}")
                return "⚠️ **Hệ thống AI đang bị quá tải (vượt giới hạn 5 câu hỏi/phút).**\nVui lòng chờ khoảng 1 phút rồi đặt câu hỏi lại để tiếp tục nhận tư vấn nhé!"
            logger.error(f"Gemini API Error: {e}", exc_info=True)
            # Fallback to LM Studio

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
            # Fallback cuối cùng: Trả về trực tiếp thông tin từ RAG nếu AI quá tải
            logger.error(f"Chat API Error: {e}", exc_info=True)
            if not context:
                return "Rất tiếc, hệ thống chưa tìm thấy thông tin kỹ thuật cụ thể. Bà con có thể thử hỏi lại với tên loại cây cụ thể (VD: bệnh rỉ sắt trên cà phê) nhé!"
                
            clean_context = context.replace(" . . .", "").replace("...", "").strip()
            clean_context = re.sub(r'\s+', ' ', clean_context)
            if len(clean_context) > 400:
                clean_context = clean_context[:400] + "..."
            return f"⚠️ AI đang bận, tôi xin trích dẫn một đoạn ngắn từ tài liệu kỹ thuật:\n\n> {clean_context}\n\n*(Lưu ý: Do AI đang bận, đây là dữ liệu thô chưa qua xử lý nên có thể khó đọc. Vui lòng thử lại sau 1 phút)*"
    else:
        logger.error("No LLM client configured!")
        if context:
            clean_context = context.replace(" . . .", "").replace("...", "").strip()
            clean_context = re.sub(r'\s+', ' ', clean_context)
            if len(clean_context) > 400:
                clean_context = clean_context[:400] + "..."
            return f"⚠️ AI đang bận, tôi xin trích dẫn một đoạn ngắn từ tài liệu kỹ thuật:\n\n> {clean_context}\n\n*(Lưu ý: Do AI đang bận, đây là dữ liệu thô chưa qua xử lý nên có thể khó đọc. Vui lòng thử lại sau 1 phút)*"
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

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    try:
        message = body.message.strip()
        if not message: return ChatResponse(answer="Bạn cần hỗ trợ gì không?", suggestions=["Kỹ thuật bón phân?", "Giá sầu riêng?"])
        
        msg_low = message.lower()
        forbidden = ["chính trị", "tôn giáo", "mã nguồn", "password"]
        if any(f in msg_low for f in forbidden):
            return ChatResponse(
                answer="Hệ thống chỉ hỗ trợ các câu hỏi liên quan đến nông nghiệp. Cảm ơn bạn.",
                suggestions=["Kỹ thuật trồng sầu riêng", "Giá cà phê hôm nay"]
            )

        intent = classify_intent(message)
        
        if intent == "greeting":
            answer = "Chào bạn! Tôi là hệ thống hỗ trợ nông nghiệp. Bạn cần tìm hiểu về kỹ thuật hay giá cả nông sản không?"
            suggestions = ["Chi phí trồng sầu riêng?", "Kỹ thuật bón phân?"]
        elif intent == "thanks":
            answer = "Rất vui được hỗ trợ bạn! Chúc bạn mùa màng thuận lợi. 🌾"
            suggestions = ["Hỏi về sâu bệnh", "Hỏi về giá cả"]
        elif intent == "finance":
            crop = detect_crop_in_message(message) or (body.context.crop if body.context else "Sầu riêng Ri6")
            loc = body.context.location if body.context else "Phường B'Lao"
            cap = body.context.capital if body.context and body.context.capital else 200_000_000
            area = body.context.area_ha if body.context and body.context.area_ha else 1.0
            mode = body.context.mode if body.context and body.context.mode else "Kinh doanh"
            answer = _run_finance_simulation(crop, cap, area, loc, mode)
            suggestions = ["Rủi ro thời tiết?", "Kỹ thuật chăm sóc?"]
        else:
            answer = await _get_rag_response(message)
            suggestions = ["Bón phân thế nào?", "Phòng trừ sâu bệnh?"]
        
        new_chat = models.ChatHistory(
            user_id=current_user.id,
            question=message,
            answer=answer
        )
        db.add(new_chat)
        db.commit()
        
        return ChatResponse(answer=answer, suggestions=suggestions)
    except Exception as e:
        logger.error(f"Chat Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi hệ thống")

@router.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    """Endpoint hỗ trợ Streaming response."""
    try:
        message = body.message.strip()
        if not message:
            return StreamingResponse(iter(["Bạn cần tôi giúp gì?"]), media_type="text/plain")
            
        intent = classify_intent(message)
        
        # Với Intent greeting/thanks/finance, chúng ta trả về 1 chunk duy nhất (vì nó nhanh)
        if intent == "greeting":
            return StreamingResponse(iter(["Chào bạn! Tôi có thể giúp gì cho mùa vụ của bạn?"]), media_type="text/plain")
        elif intent == "thanks":
            return StreamingResponse(iter(["Rất vui được giúp bà con!"]), media_type="text/plain")
        elif intent == "finance":
            # Logic finance hiện tại chưa hỗ trợ stream vì nó tính toán local nhanh
            crop = detect_crop_in_message(message) or (body.context.crop if body.context else "Sầu riêng Ri6")
            loc = body.context.location if body.context else "Phường B'Lao"
            cap = body.context.capital if body.context and body.context.capital else 200_000_000
            area = body.context.area_ha if body.context and body.context.area_ha else 1.0
            mode = body.context.mode if body.context and body.context.mode else "Kinh doanh"
            answer = _run_finance_simulation(crop, cap, area, loc, mode)
            return StreamingResponse(iter([answer]), media_type="text/plain")
            
        # Với Intent agriculture, chúng ta thực hiện Streaming từ LLM
        return StreamingResponse(_stream_rag_response(message), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Stream Error: {e}")
        return StreamingResponse(iter(["Lỗi kết nối server."]), media_type="text/plain")

@router.get("/chat/history")
async def get_chat_history(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    history = db.query(models.ChatHistory).filter(
        models.ChatHistory.user_id == current_user.id
    ).order_by(models.ChatHistory.created_at.asc()).limit(50).all()
    return history
