from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import database, models
from schemas import PredictRequest, PredictResponse
from routers.auth import get_current_user
from config import LOCATION_MAPPING, CROP_MAPPING, CROP_ALTERNATIVES, ACTION_CHECKLIST
from ml.inference import predict_risk, predict_price, get_weather
from ml.decision_engine import run_decision_engine
from ml.expert_rules import run_expert_check
import numpy as np
from limiter import limiter

router = APIRouter(
    prefix="/api",
    tags=["Prediction"],
)

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Phân tích AI dự báo canh tác & giá nông sản",
)
@limiter.limit("60/minute")
async def predict(request: Request, body: PredictRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    """
    Luồng xử lý chính:
    1. Kiểm tra rủi ro (Module 2).
    2. Nếu rủi ro cao (level 2) -> Short-circuit: Trả về cảnh báo và cây thay thế.
    3. Nếu rủi ro thấp (< 2) -> Dự báo giá (Module 1) và Decision Engine (Module 3).
    """
    # 1. Tra cứu thông tin vùng
    loc_name = body.location
    if loc_name not in LOCATION_MAPPING:
        raise HTTPException(status_code=422, detail="Vùng canh tác không hợp lệ")
    
    loc_info = LOCATION_MAPPING[loc_name]
    weather = get_weather(loc_name) # Lấy dữ liệu thật từ Open-Meteo
    
    # 2. ── EXPERT RULES: Kiểm tra Luật Sinh thái Cứng TRƯỚC Random Forest ──
    expert = run_expert_check(
        crop_name=body.crop,
        location_name=loc_name,
        elevation=loc_info["elevation"],
        temp_max=weather["temp_max"],
        temp_min=weather["temp_min"],
        precipitation=weather["precipitation"],
    )
    
    # Nếu vi phạm sinh thái nghiêm trọng (ĐỎ 🔴 risk_level == 2) → Short-circuit ngay
    if expert["ecology_violation"] and expert["ecology_violation"]["risk_level"] == 2:
        violation = expert["ecology_violation"]
        alts = [c for c in CROP_ALTERNATIVES.get(loc_name, []) if c != body.crop]
        
        return PredictResponse(
            status="success",
            location_info={
                "name": loc_name,
                "elevation": loc_info["elevation"],
                "climate": loc_info["climate"],
                "current_temp": weather.get("temp_current", weather["temp_max"]),
                "recent_rainfall_mm": weather.get("precipitation_current", weather["precipitation"])
            },
            action_plan={
                "level": violation["level"],
                "risk_score": float(violation["risk_score"]),
                "message": violation["message"],
                "reasoning": violation["reasoning"] + "\n\n💡 " + violation["suggestion"],
                "checklist": []
            },
            crop_alternatives=alts
        )
    
    # 3. Đánh giá rủi ro bằng Random Forest (Module 2) – chỉ khi ecology OK
    risk_level, risk_proba = predict_risk(loc_name, body.crop)
    
    # Nếu có vi phạm sinh thái mức VÀNG 🟡 (warning) → nâng risk_level
    if expert["ecology_violation"] and expert["ecology_violation"]["risk_level"] == 1:
        risk_level = max(risk_level, 1)
        risk_proba = max(risk_proba, expert["ecology_violation"]["risk_score"])
    
    # Mapping risk level to string
    level_map = {0: "safe", 1: "warning", 2: "danger"}
    risk_level_str = level_map.get(risk_level, "safe")
    
    # Sinh message rủi ro
    risk_message = "Điều kiện canh tác an toàn."
    if expert["ecology_violation"]:
        # Dùng message từ expert rules (chính xác hơn)
        risk_message = expert["ecology_violation"]["message"]
    elif risk_level == 2:
        risk_message = f"CẢNH BÁO: {body.crop} gặp rủi ro cao tại {loc_name}. Vui lòng xem xét cây trồng thay thế."
    elif risk_level == 1:
        risk_message = "CHÚ Ý: Có rủi ro thời tiết trung bình. Cần theo dõi sát sao."
        
    # Sinh checklist dựa trên thời tiết (cơ bản)
    checklist = []
    if weather['precipitation'] > 200:
        checklist.extend(ACTION_CHECKLIST['heavy_rain'])
    elif weather['temp_min'] < 5:
        checklist.extend(ACTION_CHECKLIST['frost'])
    else:
        checklist.extend(ACTION_CHECKLIST['normal'])

    # Short-circuit cho risk_level == 2 từ Random Forest
    if risk_level == 2:
        alts = [c for c in CROP_ALTERNATIVES.get(loc_name, []) if c != body.crop]
        
        return PredictResponse(
            status="success",
            location_info={
                "name": loc_name,
                "elevation": loc_info["elevation"],
                "climate": loc_info["climate"],
                "current_temp": weather.get("temp_current", weather["temp_max"]),
                "recent_rainfall_mm": weather.get("precipitation_current", weather["precipitation"])
            },
            action_plan={
                "level": "danger",
                "risk_score": float(risk_proba),
                "message": risk_message,
                "reasoning": f"Phân tích AI dựa trên độ cao ({loc_info['elevation']}m) và thời tiết hiện tại cho thấy rủi ro vượt ngưỡng an toàn.",
                "checklist": []
            },
            crop_alternatives=alts
        )

    # 4. Dự báo giá (Module 1) - Chạy khi Risk < 2
    # Lấy giá thực tế mới nhất từ CSV thay vì hardcode
    import pandas as pd
    import os
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    csv_map = {
        "Sầu riêng Ri6": "raw_durian_ri6.csv",
        "Chè Ô Long": "raw_oolong.csv"
    }
    
    cur_price = 50000 # Default fallback
    if body.crop in csv_map:
        csv_path = os.path.join(data_dir, csv_map[body.crop])
        if os.path.exists(csv_path):
            df_recent = pd.read_csv(csv_path)
            if not df_recent.empty:
                cur_price = float(df_recent.sort_values(by='date').iloc[-1]['price_vnd'])
    elif "Cà phê" in body.crop:
        # Giả định giá cà phê hiện tại (có thể fetch từ Yahoo Finance trong thực tế)
        cur_price = 120000 if "Robusta" in body.crop else 150000
    
    forecast_data = predict_price(body.crop, cur_price, loc_name)
    
    # Tính toán trend
    pred_30d = forecast_data[-1]["predicted"]
    trend_pct = ((pred_30d - cur_price) / cur_price) * 100
    trend_str = "up" if trend_pct > 0.5 else "down" if trend_pct < -0.5 else "stable"
    
    # 5. Chạy Decision Engine (Module 3)
    decision = run_decision_engine(
        crop_name=body.crop,
        risk_level=risk_level,
        current_price=cur_price,
        predicted_price=pred_30d,
        capital=body.capital,
        area_ha=body.area_ha,
        temp_min=weather["temp_min"],
        precipitation=weather["precipitation"],
        mode=body.mode,
        ecology_violation=expert["ecology_violation"]  # ★ Truyền vi phạm sinh thái để kích hoạt Price Penalty
    )
    
    # ── Sinh reasoning bám vào số liệu thực tế (Explainable AI) ──
    fa = decision["financial_analysis"]
    roi = fa["roi_pct"]
    
    # Đánh giá nhiệt độ cho cây trồng (có tham chiếu expert rules)
    temp_assessment = ""
    if weather["temp_max"] >= 25 and weather["temp_max"] <= 32:
        temp_assessment = f"nằm trong ngưỡng sinh thái lý tưởng cho {body.crop}"
    elif weather["temp_max"] > 32:
        temp_assessment = f"hơi cao, cần chú ý tưới mát cho {body.crop}"
    else:
        temp_assessment = f"đạt mức chấp nhận được cho {body.crop}"
    
    # Đánh giá lượng mưa
    rain_assessment = ""
    if weather["precipitation"] < 20:
        rain_assessment = "không gây úng rễ"
    elif weather["precipitation"] < 100:
        rain_assessment = "ở mức trung bình, cần theo dõi thoát nước"
    else:
        rain_assessment = "khá cao, cần khơi thông rãnh thoát nước"
    
    # Đánh giá xu hướng giá
    price_direction = "tăng" if trend_pct > 0 else "giảm" if trend_pct < 0 else "ổn định"
    formatted_cur_price = f"{cur_price:,.0f}".replace(",", ".")
    formatted_pred_price = f"{pred_30d:,.0f}".replace(",", ".")
    
    # ── Đánh giá độ cao + sinh thái (KHÔNG hardcode "phù hợp") ──
    # Kiểm tra xem cây trồng có nằm trong danh sách phù hợp của vùng không
    suitable_crops = loc_info.get("suitable_crops", [])
    if body.crop in suitable_crops:
        elevation_assessment = f"Độ cao {loc_info['elevation']}m phù hợp với sinh thái của {body.crop}"
    else:
        # Có vi phạm ecology nhưng chỉ ở mức warning (đã qua short-circuit danger)
        elevation_assessment = f"Độ cao {loc_info['elevation']}m KHÔNG LÝ TƯỞNG cho {body.crop} – cần lưu ý rủi ro"
    
    # Nếu có ecology warning từ expert rules, thêm vào reasoning
    ecology_note = ""
    if expert["ecology_violation"]:
        ecology_note = f"\n\n⚠️ {expert['ecology_violation']['message']}\n💡 {expert['ecology_violation']['suggestion']}"
    
    # ── Format reasoning có xuống dòng rõ ràng từng ý ──
    # Kiểm tra xem có price penalty từ vi phạm sinh thái không
    price_penalty_factor = fa.get("price_penalty_factor", 1.0)
    price_penalty_applied = fa.get("price_penalty_applied", False)
    
    if price_penalty_applied and price_penalty_factor < 1.0:
        # Giá thu mua thực tế bị giáng cấp
        actual_cur_price = cur_price * price_penalty_factor
        actual_pred_price = pred_30d * price_penalty_factor
        formatted_actual_cur = f"{actual_cur_price:,.0f}".replace(",", ".")
        formatted_actual_pred = f"{actual_pred_price:,.0f}".replace(",", ".")
        
        price_line = (
            f"⚠️ Cảnh báo Giá thu mua: Dù giá thị trường của {body.crop} chuẩn là "
            f"{formatted_cur_price}đ/kg, nhưng do trồng sai vùng ({loc_name}), "
            f"sản phẩm bị giáng cấp. Giá thu mua thực tế của bạn bị ép xuống chỉ còn "
            f"~{formatted_actual_cur} VNĐ/kg."
        )
    else:
        price_line = (
            f"📈 Giá dự báo {price_direction} {abs(trend_pct):.1f}% trong 30 ngày "
            f"(từ {formatted_cur_price} → {formatted_pred_price} VNĐ/kg)."
        )
    
    detailed_reasoning = (
        f"🌡️ Nhiệt độ trung bình {weather['temp_max']}°C {temp_assessment}.\n"
        f"🌧️ Lượng mưa {weather['precipitation']}mm/ngày {rain_assessment}.\n"
        f"⛰️ {elevation_assessment}.\n"
        f"{price_line}\n"
        f"💰 ROI kỳ vọng: {roi:.1f}%."
        f"{ecology_note}"
    )
    
    # ── Sinh lời khuyên giá theo mode ──
    if body.mode == "Kiến thiết":
        price_advice = (
            f"⏳ Khuyến cáo: {body.crop} cần 3–5 năm mới cho thu hoạch. "
            f"Dự báo giá ngắn hạn không áp dụng cho giai đoạn này. "
            f"Vui lòng tham khảo Phân tích Tài chính dài hạn bên cạnh."
        )
    else:
        # Mode Kinh doanh: sinh text gợi ý chốt giá
        if trend_pct > 0:
            price_advice = (
                f"📈 Giá {body.crop} dự báo tăng {abs(trend_pct):.1f}% vào tháng tới. "
                f"Nông dân cân nhắc thời điểm chốt giá bán hoặc trữ kho để tối ưu lợi nhuận."
            )
        elif trend_pct < -0.5:
            price_advice = (
                f"📉 Giá {body.crop} dự báo giảm {abs(trend_pct):.1f}% trong 30 ngày tới. "
                f"Nên cân nhắc bán sớm hoặc chế biến sâu để giữ giá trị sản phẩm."
            )
        else:
            price_advice = (
                f"📊 Giá {body.crop} dự báo ổn định trong 30 ngày tới. "
                f"Thị trường không có biến động lớn, có thể bán hàng theo kế hoạch."
            )
    
    # ── Sinh checklist cụ thể theo cây trồng + thời tiết ──
    specific_checklist = []
    
    # ── EXPERT RULES: Thêm cảnh báo thời tiết khẩn cấp từ bộ luật chuyên gia ──
    if expert["weather_warnings"]:
        for warning in expert["weather_warnings"]:
            # Thêm tiêu đề cảnh báo như một action đặc biệt
            specific_checklist.append({
                "action": f"{warning['title']}: {warning['description']}",
                "done": False
            })
            # Thêm tất cả actions khẩn cấp
            specific_checklist.extend(warning["actions"])
    
    # Checklist dựa trên thời tiết (cơ bản, chỉ thêm nếu chưa có expert warning)
    if not expert["weather_warnings"]:
        if weather['precipitation'] > 200:
            specific_checklist.extend(ACTION_CHECKLIST['heavy_rain'])
        elif weather['temp_min'] < 5:
            specific_checklist.extend(ACTION_CHECKLIST['frost'])
        elif weather['temp_min'] <= 10:
            specific_checklist.extend(ACTION_CHECKLIST['cold'])
    
    # ── Xử lý logic bón phân NPK theo lượng mưa ──
    if weather['precipitation'] >= 15:
        npk_action = f"Độ ẩm đất tốt ({weather['precipitation']}mm) – tiến hành bón thúc NPK"
    else:
        temp_warning = " Nhiệt độ >30°C chú ý tưới giữ ẩm." if weather['temp_max'] > 30 else ""
        npk_action = f"Đất khô (mưa {weather['precipitation']}mm quá ít), KHÔNG rải NPK trừ khi có hệ thống tưới béc hỗ trợ.{temp_warning}"

    # Checklist cụ thể theo cây trồng + tình huống
    crop_specific = {
        "Cà phê Robusta": [
            {"action": npk_action, "done": False},
            {"action": f"Nhiệt độ {weather['temp_max']}°C lý tưởng – theo dõi ra hoa đợt phụ", "done": False},
            {"action": "Phun phòng rỉ sắt nếu ẩm độ không khí > 85%", "done": False},
        ],
        "Cà phê Arabica": [
            {"action": f"Nhiệt {weather['temp_min']}–{weather['temp_max']}°C – kiểm tra tỷ lệ đậu trái", "done": False},
            {"action": f"Lượng mưa {weather['precipitation']}mm – đủ ẩm, tạm ngưng tưới bổ sung" if weather['precipitation'] > 20 else "Lượng mưa thấp, cần theo dõi độ ẩm và tưới bổ sung", "done": False},
            {"action": "Bón phân kali tăng cường sức đề kháng sương muối" if weather["temp_min"] <= 10 else "Phun thuốc gốc Đồng (Copper) phòng trừ nấm nếu ẩm độ cao", "done": False},
        ],
        "Sầu riêng Ri6": [
            {"action": f"Nhiệt độ {weather['temp_max']}°C tốt – kiểm tra tình trạng xổ nhụy", "done": False},
            {"action": f"Lượng mưa {weather['precipitation']}mm – kiểm tra hệ thống thoát nước gốc", "done": False},
            {"action": "Phun canxi-bo qua lá nếu đang giai đoạn nuôi trái", "done": False},
        ],
        "Chè Ô Long": [
            {"action": f"Nhiệt {weather['temp_max']}°C lý tưởng cho đâm búp – chuẩn bị thu hoạch đợt mới", "done": False},
            {"action": npk_action, "done": False},
            {"action": "Kiểm tra mật độ bọ cánh tơ trên búp non", "done": False},
        ],
    }
    
    specific_checklist.extend(crop_specific.get(body.crop, ACTION_CHECKLIST['normal']))
    
    # ── Xử lý forecast theo mode ──
    if body.mode == "Kiến thiết":
        # Trồng mới: KHÔNG trả forecast giá ngắn hạn
        price_forecast_data = None
    else:
        # Nếu có price penalty → điều chỉnh biểu đồ giá xuống giá thu mua thực tế
        if price_penalty_applied and price_penalty_factor < 1.0:
            adjusted_forecast = []
            for point in forecast_data:
                adjusted_forecast.append({
                    "date": point["date"],
                    "predicted": point["predicted"] * price_penalty_factor,
                    "min": point["min"] * price_penalty_factor,
                    "max": point["max"] * price_penalty_factor,
                })
            adjusted_cur_price = cur_price * price_penalty_factor
            adjusted_pred_30d = pred_30d * price_penalty_factor
            adjusted_trend_pct = ((adjusted_pred_30d - adjusted_cur_price) / adjusted_cur_price) * 100 if adjusted_cur_price > 0 else 0
            adjusted_trend_str = "up" if adjusted_trend_pct > 0.5 else "down" if adjusted_trend_pct < -0.5 else "stable"
            
            price_forecast_data = {
                "crop": body.crop,
                "unit": "VND/kg (giá thu mua thực tế - đã giáng cấp)",
                "current_price": adjusted_cur_price,
                "trend": adjusted_trend_str,
                "trend_pct": adjusted_trend_pct,
                "forecast_30_days": adjusted_forecast,
                "price_advice": price_advice
            }
        else:
            price_forecast_data = {
                "crop": body.crop,
                "unit": "VND/kg",
                "current_price": cur_price,
                "trend": trend_str,
                "trend_pct": trend_pct,
                "forecast_30_days": forecast_data,
                "price_advice": price_advice
            }
    
    # ── Cảnh báo giá vật tư phân bón ──
    fertilizer_advice = "Dự báo giá phân bón (Ure, DAP) có thể biến động nhẹ trong tháng tới do ảnh hưởng của giá nguyên liệu đầu vào. Bà con nên theo dõi thông tin từ đại lý địa phương để chủ động nguồn phân bón cho vụ mùa, tránh mua gom tích trữ quá mức."
    
    # 6. Lưu vào lịch sử tra cứu
    try:
        new_history = models.SearchHistory(
            user_id=current_user.id,
            location=loc_name,
            crop=body.crop,
            mode=body.mode,
            capital=body.capital,
            area_ha=body.area_ha,
            risk_level=risk_level_str,
            recommendation=decision["production_decision"]["recommendation"]
        )
        db.add(new_history)
        db.commit()
    except Exception as e:
        print(f"Error saving search history: {e}")

    return PredictResponse(
        status="success",
        location_info={
            "name": loc_name,
            "elevation": loc_info["elevation"],
            "climate": loc_info["climate"],
            "current_temp": weather.get("temp_current", weather["temp_max"]),
            "recent_rainfall_mm": weather.get("precipitation_current", weather["precipitation"])
        },
        action_plan={
            "level": risk_level_str,
            "risk_score": float(risk_proba),
            "message": risk_message,
            "reasoning": detailed_reasoning,
            "checklist": specific_checklist
        },
        price_forecast=price_forecast_data,
        production_decision={
            "recommendation": decision["production_decision"]["recommendation"],
            "confidence": decision["production_decision"]["confidence"],
            "reasoning": decision["production_decision"]["reasoning"]
        },
        financial_analysis=decision["financial_analysis"],
        farming_mode=body.mode,
        price_advice=price_advice,
        fertilizer_advice=fertilizer_advice
    )
