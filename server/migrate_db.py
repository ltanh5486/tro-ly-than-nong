from database import engine
from models import Base


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database schema is ready.")


if __name__ == "__main__":
    main()
