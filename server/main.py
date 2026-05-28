"""
Tro Ly Than Nong - API Server
"""

import sys
import os
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import models
from database import engine
from fastapi.staticfiles import StaticFiles

# Encoding fix for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from database import SessionLocal
import bcrypt

APP_ENV = os.getenv("APP_ENV", "development").lower()

def init_admin():
    db = SessionLocal()
    try:
        admin_username = "admin"
        existing = db.query(models.User).filter(models.User.username == admin_username).first()
        if not existing:
            salt = bcrypt.gensalt()
            default_pwd = os.getenv("DEFAULT_ADMIN_PWD")
            if not default_pwd:
                raise RuntimeError("DEFAULT_ADMIN_PWD must be set before creating the default admin account.")
            hashed = bcrypt.hashpw(default_pwd.encode('utf-8'), salt).decode('utf-8')
            new_admin = models.User(
                username=admin_username,
                email="admin@thannong.ai",
                full_name="Administrator",
                hashed_password=hashed,
                role="admin",
                is_active=True
            )
            db.add(new_admin)
            db.commit()
            logger.info("System: Created default admin account.")

        # Seed lịch sử tìm kiếm mẫu để biểu đồ hiển thị sinh động và đủ 4 loại cây trồng
        history_count = db.query(models.SearchHistory).count()
        if history_count == 0:
            admin_user = db.query(models.User).filter(models.User.role == "admin").first()
            admin_id = admin_user.id if admin_user else 1

            default_history = [
                models.SearchHistory(
                    user_id=admin_id,
                    location="B'Lao, Bảo Lộc",
                    crop="Chè Ô Long",
                    mode="Tối ưu hóa lợi nhuận",
                    capital=150000000.0,
                    area_ha=2.5,
                    risk_level="Thấp",
                    recommendation="Khuyến nghị tập trung bón phân hữu cơ và tỉa cành định kỳ."
                ),
                models.SearchHistory(
                    user_id=admin_id,
                    location="Lộc Thanh, Bảo Lộc",
                    crop="Cà phê Robusta",
                    mode="Tối ưu hóa lợi nhuận",
                    capital=100000000.0,
                    area_ha=1.8,
                    risk_level="Trung bình",
                    recommendation="Đề phòng bệnh rỉ sắt trong giai đoạn mùa mưa sắp tới."
                ),
                models.SearchHistory(
                    user_id=admin_id,
                    location="Đại Lào, Bảo Lộc",
                    crop="Sầu riêng Ri6",
                    mode="Phòng ngừa rủi ro",
                    capital=300000000.0,
                    area_ha=1.2,
                    risk_level="Cao",
                    recommendation="Đặc biệt chú ý thoát nước tốt để tránh thối rễ."
                ),
                models.SearchHistory(
                    user_id=admin_id,
                    location="B'Lao, Bảo Lộc",
                    crop="Cà phê Arabica",
                    mode="Tối ưu hóa lợi nhuận",
                    capital=80000000.0,
                    area_ha=1.0,
                    risk_level="Thấp",
                    recommendation="Khuyến nghị thu hoạch đúng độ chín để nâng cao chất lượng hạt Arabica Catimor."
                )
            ]
            for h in default_history:
                db.add(h)
            db.commit()
            logger.info("System: Seeded search history with 4 main crops.")

    except Exception as e:
        logger.error(f"Error initializing admin: {e}")
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application... Initializing database.")
    models.Base.metadata.create_all(bind=engine)
    init_admin()
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="Trợ Lý Thần Nông",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost",
        "http://127.0.0.1"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from routers import predict, chat, auth, admin

def get_real_ip(request: Request):
    x_forwarded = request.headers.get("X-Forwarded-For")
    if x_forwarded:
        return x_forwarded.split(",")[0].strip()
    return request.client.host

# Rate limiting
from limiter import limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routers
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(admin.router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


@app.get("/", tags=["Health"])
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/index.html")


# Static files
client_path = os.path.join(os.path.dirname(current_dir), "client")
if os.path.exists(client_path):
    app.mount("/", StaticFiles(directory=client_path, html=True), name="client")
