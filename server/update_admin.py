import os

from database import SessionLocal
from models import User


def main() -> None:
    admin_email = os.getenv("ADMIN_EMAIL", "admin@thannong.ai")
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            print("Error: Admin user not found")
            return
        admin.email = admin_email
        db.commit()
        print(f"Success: Updated admin email to {admin_email}")
    except Exception as exc:
        db.rollback()
        print(f"Error: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
