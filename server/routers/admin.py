
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session
import database, models, schemas
from routers.auth import get_current_user
import os
import subprocess

router = APIRouter(prefix="/api/admin", tags=["Admin"])

def check_admin(current_user: models.User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền truy cập chức năng này."
        )
    return current_user

@router.get("/stats")
def get_stats(db: Session = Depends(database.get_db), admin: models.User = Depends(check_admin)):
    """Lấy thống kê tổng quan và dữ liệu biểu đồ cho Admin."""
    total_users = db.query(models.User).count()
    farmers = db.query(models.User).filter(models.User.role == "farmer").count()
    admins = db.query(models.User).filter(models.User.role == "admin").count()
    
    # Thống kê cơ cấu cây trồng (cho biểu đồ tròn)
    crop_stats = db.query(
        models.SearchHistory.crop, 
        func.count(models.SearchHistory.id).label('count')
    ).group_by(models.SearchHistory.crop).all()
    
    # Bộ ánh xạ chuẩn hóa về đúng 4 loại cây trồng chính
    crop_mapping = {
        "chè oolong": "Chè Ô Long",
        "chè ô long": "Chè Ô Long",
        "oolong": "Chè Ô Long",
        "cà phê robusta": "Cà phê Robusta",
        "cà phê robusta (tr4/tr9)": "Cà phê Robusta",
        "robusta": "Cà phê Robusta",
        "cà phê arabica": "Cà phê Arabica",
        "cà phê arabica (catimor)": "Cà phê Arabica",
        "arabica": "Cà phê Arabica",
        "sầu riêng ri6": "Sầu riêng Ri6",
        "sầu riêng": "Sầu riêng Ri6",
        "durian": "Sầu riêng Ri6"
    }

    normalized_crops = {
        "Chè Ô Long": 0,
        "Cà phê Robusta": 0,
        "Cà phê Arabica": 0,
        "Sầu riêng Ri6": 0
    }
    for crop_name, count in crop_stats:
        if not crop_name:
            continue
        name_clean = crop_name.lower().strip()
        norm_name = crop_mapping.get(name_clean, crop_name.strip())
        if norm_name in normalized_crops:
            normalized_crops[norm_name] += count
        else:
            normalized_crops[norm_name] = count

    crop_distribution = [{"label": label, "value": val} for label, val in normalized_crops.items()]

    # Thống kê phân bố vùng trồng (cho biểu đồ cột)
    region_stats = db.query(
        models.SearchHistory.location, 
        func.count(models.SearchHistory.id).label('count')
    ).group_by(models.SearchHistory.location).all()
    
    return {
        "total_users": total_users,
        "farmers": farmers,
        "admins": admins,
        "crop_distribution": crop_distribution,
        "region_distribution": [{"label": r[0], "value": r[1]} for r in region_stats],
        "kb_status": "Ready" if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")) else "Not Initialized"
    }

@router.post("/ingest")
def trigger_ingest(admin: models.User = Depends(check_admin)):
    """Kích hoạt quá trình nạp lại tri thức RAG."""
    try:
        # Chạy script ingest trong một tiến trình riêng
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ingest_kb.py")
        subprocess.Popen(["python", script_path])
        return {"message": "Đang bắt đầu quá trình nạp tri thức ngầm..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khởi động: {str(e)}")

@router.get("/users", response_model=list[schemas.User])
def list_users(db: Session = Depends(database.get_db), admin: models.User = Depends(check_admin)):
    """Danh sách người dùng."""
    return db.query(models.User).all()

@router.get("/files")
def list_knowledge_files(admin: models.User = Depends(check_admin)):
    """Liệt kê danh sách file PDF trong kho tri thức."""
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base")
    if not os.path.exists(kb_path):
        return []
    
    files = [f for f in os.listdir(kb_path) if f.endswith(".pdf")]
    file_list = []
    for f in files:
        f_path = os.path.join(kb_path, f)
        f_stats = os.stat(f_path)
        file_list.append({
            "name": f,
            "size": f_stats.st_size,
            "created_at": f_stats.st_ctime
        })
    return file_list

from fastapi import File, UploadFile
import shutil

@router.post("/upload")
async def upload_document(file: UploadFile = File(...), admin: models.User = Depends(check_admin)):
    """Tải lên tài liệu PDF mới."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file định dạng PDF.")
    
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base")
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
    
    safe_filename = os.path.basename(file.filename)
    file_location = os.path.join(kb_path, safe_filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"message": f"Đã tải lên thành công: {safe_filename}", "filename": safe_filename}

@router.delete("/files/{filename}")
def delete_document(filename: str, admin: models.User = Depends(check_admin)):
    """Xóa tài liệu khỏi kho tri thức."""
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base")
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(kb_path, safe_filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Đã xóa file: {filename}"}
    else:
        raise HTTPException(status_code=404, detail="Không tìm thấy file.")

@router.get("/trends")
def get_chat_trends(db: Session = Depends(database.get_db), admin: models.User = Depends(check_admin)):
    """Phân tích các từ khóa phổ biến trong câu hỏi của người dùng."""
    chats = db.query(models.ChatHistory.question)\
              .order_by(models.ChatHistory.created_at.desc())\
              .limit(2000)\
              .all()
    all_text = " ".join([c[0].lower() for c in chats if c[0]])
    
    # Các từ khóa nông nghiệp cần theo dõi
    keywords = ["sầu riêng", "cà phê", "arabica", "robusta", "bón phân", "sâu bệnh", "giá", "chi phí", "thu hoạch", "tưới nước"]
    trends = []
    for kw in keywords:
        count = all_text.count(kw)
        if count > 0:
            trends.append({"topic": kw.capitalize(), "count": count})
    
    # Sắp xếp theo số lượng giảm dần
    trends = sorted(trends, key=lambda x: x["count"], reverse=True)
    return trends[:5]
