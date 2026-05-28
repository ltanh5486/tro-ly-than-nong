import sys
import os
from pathlib import Path

# Xác định đường dẫn tuyệt đối của thư mục gốc và thư mục server
BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR / "server"

# Thêm thư mục server vào sys.path để Python tìm thấy database.py và models.py
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

# Fix encoding cho Windows để tránh lỗi hiển thị emoji/tiếng Việt
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import từ package server
from server.database import SessionLocal
from server import models
import bcrypt

def create_admin():
    db = SessionLocal()
    try:
        admin_password = os.getenv("DEFAULT_ADMIN_PWD")
        if not admin_password:
            raise RuntimeError("Set DEFAULT_ADMIN_PWD before creating or resetting the admin account.")

        # Kiểm tra xem admin đã tồn tại chưa
        admin_username = "admin"
        admin_email = "admin@thannong.ai"
        existing_admin = db.query(models.User).filter(models.User.username == admin_username).first()
        
        if existing_admin:
            print(f"Tài khoản {admin_username} đã tồn tại. Đang cập nhật mật khẩu và quyền admin...")
            existing_admin.role = "admin"
            existing_admin.email = admin_email
            salt = bcrypt.gensalt()
            existing_admin.hashed_password = bcrypt.hashpw(admin_password.encode('utf-8'), salt).decode('utf-8')
            db.commit()
            print("Cập nhật thành công!")
        else:
            print(f"Đang tạo tài khoản admin mới: {admin_username}")
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(admin_password.encode('utf-8'), salt).decode('utf-8')
            
            new_admin = models.User(
                username=admin_username,
                email=admin_email,
                full_name="Administrator",
                hashed_password=hashed_password,
                role="admin",
                is_active=True
            )
            db.add(new_admin)
            db.commit()
            print("Tạo tài khoản Admin thành công!")
            
    except Exception as e:
        print(f"Lỗi: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_admin()
