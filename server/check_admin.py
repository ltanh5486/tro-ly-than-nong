from database import SessionLocal
from models import User


def main() -> None:
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if admin:
            print(f"User: {admin.username}, Email: {admin.email}")
        else:
            print("Khong tim thay tai khoan admin")
    finally:
        db.close()


if __name__ == "__main__":
    main()
