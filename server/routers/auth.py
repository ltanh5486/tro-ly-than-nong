from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
import bcrypt
from typing import Optional
import os
from dotenv import load_dotenv

import secrets
import schemas, models, database
from mail_service import send_otp_email
from limiter import limiter

load_dotenv()

# Cấu hình bảo mật
APP_ENV = os.getenv("APP_ENV", "development").lower()
SECRET_KEY_ENV = os.getenv("SECRET_KEY")
SECRET_KEY = SECRET_KEY_ENV or "your-secret-key-for-dev-only"
if APP_ENV in {"production", "prod"} and not SECRET_KEY_ENV:
    raise RuntimeError("SECRET_KEY must be set in production.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 ngày

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# ── Helper Functions ──────────────────────────────────────

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ── API Endpoints ──────────────────────────────────────────

@router.post("/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    # Kiểm tra username đã tồn tại chưa
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã được sử dụng")
    
    # Kiểm tra email đã tồn tại chưa (Bắt buộc) - Chuyển về chữ thường
    email_lower = user.email.lower()
    db_email = db.query(models.User).filter(models.User.email == email_lower).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email đã được đăng ký")
    
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=email_lower,
        hashed_password=hashed_password,
        full_name=user.full_name,
        role="farmer"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    # Đăng nhập bằng username
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không chính xác",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "role": user.role,
        "full_name": user.full_name
    }

@router.post("/forgot-password")
@limiter.limit("3/10minutes")
def forgot_password(request: Request, body: schemas.ForgotPasswordRequest, db: Session = Depends(database.get_db)):
    email_lower = body.email.lower()
    user = db.query(models.User).filter(models.User.email == email_lower).first()
    if not user:
        # Bảo mật: không báo lỗi nếu email không tồn tại
        return {"message": "Nếu email tồn tại trong hệ thống, mã OTP sẽ được gửi đi."}
    
    # Tạo mã OTP 6 số an toàn bằng secrets
    otp_code = ''.join(secrets.choice("0123456789") for _ in range(6))
    user.otp_code = otp_code
    user.otp_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
    db.commit()
    
    send_otp_email(user.email, otp_code)
    return {"message": "Mã OTP đã được gửi về email của bạn."}

@router.post("/reset-password")
@limiter.limit("5/10minutes")
def reset_password(request: Request, body: schemas.ResetPasswordRequest, db: Session = Depends(database.get_db)):
    email_lower = body.email.lower()
    user = db.query(models.User).filter(models.User.email == email_lower).first()
    if not user or user.otp_code != body.otp:
        raise HTTPException(status_code=400, detail="Mã OTP không chính xác")
    
    if datetime.now(timezone.utc) > user.otp_expiry:
        raise HTTPException(status_code=400, detail="Mã OTP đã hết hạn")
    
    user.hashed_password = get_password_hash(body.new_password)
    user.otp_code = None
    user.otp_expiry = None
    db.commit()
    
    return {"message": "Đặt lại mật khẩu thành công. Vui lòng đăng nhập lại."}

# Dependency để lấy user hiện tại từ token
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin đăng nhập",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

@router.post("/change-password")
def change_password(body: schemas.ChangePasswordRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    # Kiểm tra mật khẩu cũ
    if not verify_password(body.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Mật khẩu cũ không chính xác")
    
    # Cập nhật mật khẩu mới
    current_user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    
    return {"message": "Đổi mật khẩu thành công!"}
