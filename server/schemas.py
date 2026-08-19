"""
📋 Pydantic Schemas - Validate Input/Output

Định nghĩa cấu trúc dữ liệu cho request/response API.
Đảm bảo dữ liệu vào/ra luôn đúng format.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """Schema cho request từ Frontend gửi lên API."""
    location: str = Field(
        ...,
        description="Tên vùng canh tác: 'Phường B'Lao' hoặc 'Phường Xuân Trường'",
        examples=["Phường B'Lao"],
    )
    crop: str = Field(
        ...,
        description="Tên cây trồng cụ thể",
        examples=["Cà phê Robusta"],
    )
    mode: str = Field(
        "Kinh doanh",
        description="Chế độ canh tác: 'Kinh doanh' (Thu hoạch) hoặc 'Kiến thiết' (Trồng mới)"
    )
    capital: float = Field(
        ...,
        ge=50_000_000,
        le=10_000_000_000,
        description="Vốn đầu tư (VND), từ 50 triệu đến 10 tỷ",
        examples=[100_000_000],
    )
    area_ha: float = Field(
        ...,
        ge=0.1,
        le=50.0,
        description="Diện tích canh tác (ha), tối đa 50 ha",
        examples=[1.0],
    )


# ═══════════════════════════════════════════════════════════
# OUTPUT SCHEMAS (Nested)
# ═══════════════════════════════════════════════════════════

class LocationInfo(BaseModel):
    """Thông tin vùng canh tác."""
    name: str
    elevation: int
    climate: Optional[str] = None
    current_temp: Optional[float] = None
    recent_rainfall_mm: Optional[float] = None


class ChecklistItem(BaseModel):
    """Một hành động trong checklist."""
    action: str
    done: bool = False


class ActionPlan(BaseModel):
    """Kế hoạch hành động & đánh giá rủi ro."""
    level: str = Field(
        ...,
        description="Mức rủi ro: 'safe', 'warning', 'danger'",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Điểm rủi ro từ 0.0 (an toàn) đến 1.0 (nguy hiểm)",
    )
    message: str = Field(
        ...,
        description="Giải thích rõ ràng cho nông dân VÌ SAO lại đưa ra đánh giá này",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Giải thích chi tiết các yếu tố ảnh hưởng (Explainable AI)",
    )
    checklist: List[ChecklistItem] = []


class PriceForecastPoint(BaseModel):
    """Một điểm dự báo giá."""
    date: str
    min: float
    predicted: float
    max: float


class PriceForecast(BaseModel):
    """Kết quả dự báo giá từ Module 1 (TFT/XGBoost)."""
    crop: str
    unit: str = "VND/kg"
    current_price: float
    trend: str = Field(
        ...,
        description="Xu hướng giá: 'up', 'down', 'stable'",
    )
    trend_pct: float = Field(
        ...,
        description="Phần trăm thay đổi giá dự báo",
    )
    forecast_30_days: List[PriceForecastPoint] = []
    price_advice: Optional[str] = Field(
        None,
        description="Lời khuyên giá cho nông dân dựa trên xu hướng",
    )


class ProductionDecision(BaseModel):
    """Khuyến nghị quyết định sản xuất từ Module 3."""
    recommendation: str = Field(
        ...,
        description="Kết luận chính: THUẬN LỢI / THẬN TRỌNG / TỪ CHỐI",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Độ tin cậy của khuyến nghị",
    )
    reasoning: str = Field(
        ...,
        description="Giải thích VÌ SAO đưa ra khuyến nghị này (Explainable AI)",
    )


class CostBreakdown(BaseModel):
    """Phân chia chi phí thành 3 khoản cho biểu đồ tài chính."""
    seeds: float = Field(..., description="Chi phí giống (VND)")
    fertilizer: float = Field(..., description="Chi phí phân bón (VND)")
    labor: float = Field(..., description="Chi phí nhân công (VND)")


class FinancialAnalysis(BaseModel):
    """Phân tích tài chính từ Decision Engine."""
    capital_input: float
    area_ha: float
    estimated_cost: float
    estimated_revenue: float
    estimated_profit: float
    roi_pct: float
    breakeven_price: float = Field(
        ...,
        description="Giá hòa vốn (VND/kg)",
    )
    cost_breakdown: Optional[CostBreakdown] = Field(
        None,
        description="Phân chia chi phí (giống/phân bón/nhân công)",
    )


# ═══════════════════════════════════════════════════════════
# RESPONSE SCHEMA (Tổng hợp)
# ═══════════════════════════════════════════════════════════

class PredictResponse(BaseModel):
    """Schema đầy đủ cho JSON phản hồi từ API /api/predict.

    Có 2 trường hợp:
    - risk_level == 2: Chỉ có location_info, action_plan, crop_alternatives
    - risk_level < 2: Có đầy đủ tất cả các trường
    """
    status: str = "success"
    location_info: LocationInfo
    action_plan: ActionPlan
    crop_alternatives: Optional[List[str]] = Field(
        None,
        description="Danh sách cây trồng thay thế (chỉ khi risk_level == 2)",
    )
    price_forecast: Optional[PriceForecast] = Field(
        None,
        description="Dự báo giá (null khi risk_level == 2)",
    )
    production_decision: Optional[ProductionDecision] = Field(
        None,
        description="Khuyến nghị sản xuất (null khi risk_level == 2)",
    )
    financial_analysis: Optional[FinancialAnalysis] = Field(
        None,
        description="Phân tích tài chính (null khi risk_level == 2)",
    )
    farming_mode: Optional[str] = Field(
        None,
        description="Chế độ canh tác: 'Kinh doanh' hoặc 'Kiến thiết'",
    )
    price_advice: Optional[str] = Field(
        None,
        description="Lời khuyên giá cho nông dân",
    )
    fertilizer_advice: Optional[str] = Field(
        None,
        description="Dự báo hoặc cảnh báo về giá vật tư nông nghiệp (phân bón)",
    )


# ═══════════════════════════════════════════════════════════
# CHAT SCHEMAS
# ═══════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """Schema cho một tin nhắn trong hội thoại."""
    role: str = Field(..., description="Vai trò: 'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")


class ChatContext(BaseModel):
    """Ngữ cảnh hiện tại của UI để AI biết user đang xem gì."""

    location: Optional[str] = None
    crop: Optional[str] = None
    mode: Optional[str] = "Kinh doanh"
    capital: Optional[float] = None
    area_ha: Optional[float] = None

    # Memory của ca bệnh gần nhất
    disease_context: Optional[dict] = None
    disease_session: Optional[dict] = None


class ChatRequest(BaseModel):
    """Request gửi lên API Chat."""
    message: str = Field(..., description="Câu hỏi của nông dân")
    history: List[ChatMessage] = Field(default=[], description="Lịch sử trò chuyện")
    context: Optional[ChatContext] = Field(None, description="Ngữ cảnh UI hiện tại")


class ChatResponse(BaseModel):
    """Response trả về từ API Chat."""
    status: str = "success"
    answer: str = Field(..., description="Câu trả lời từ AI")
    suggestions: List[str] = Field(default=[], description="Các câu hỏi gợi ý tiếp theo")
    tool_calls: Optional[List[dict]] = Field(None, description="Thông tin các hàm AI đã gọi (nếu có)")


# ═══════════════════════════════════════════════════════════
# AUTH & USER SCHEMAS
# ═══════════════════════════════════════════════════════════

class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: str = "farmer"  # "farmer" hoặc "admin"

class UserCreate(UserBase):
    password: str
    email: EmailStr

class User(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    full_name: Optional[str]

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginRequest(BaseModel):
    username: str  # Đăng nhập bằng username
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str
