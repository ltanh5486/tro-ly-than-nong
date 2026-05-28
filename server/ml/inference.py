import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import warnings

# Ẩn các cảnh báo lệch phiên bản thư viện để làm sạch Log
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Thêm thư mục server vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOCATION_MAPPING, CROP_MAPPING, RISK_THRESHOLDS

# Ánh xạ tên cây trồng sang file cơ sở
CROP_FILE_MAP = {
    "Cà phê Robusta": "robusta",
    "Cà phê Arabica": "arabica",
    "Sầu riêng Ri6": "durian_ri6",
    "Chè Ô Long": "oolong"
}

# Đường dẫn tới thư mục chứa models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ai_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_pkl(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _load_price_history(crop_name):
    file_base = CROP_FILE_MAP.get(crop_name)
    if not file_base:
        return None

    csv_path = os.path.join(DATA_DIR, f"processed_{file_base}.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.empty or "price_vnd" not in df.columns:
        return None

    if "date" in df.columns:
        df = df.sort_values("date")
    return df


def _smooth_fallback_forecast(current_price, base_date, horizon=30):
    forecast = []
    for day in range(1, horizon + 1):
        target_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        seasonal_adjustment = 0.005 * np.sin(day / 30 * np.pi)
        predicted = current_price * (1 + seasonal_adjustment)
        forecast.append({
            "date": target_date,
            "min": float(predicted * 0.95),
            "predicted": float(predicted),
            "max": float(predicted * 1.05),
        })
    return forecast


def _historical_price_forecast(crop_name, current_price, base_date, horizon=30):
    """Deterministic statistical forecast from recent price history."""
    df = _load_price_history(crop_name)
    if df is None or len(df) < 10:
        return _smooth_fallback_forecast(current_price, base_date, horizon)

    prices = pd.to_numeric(df["price_vnd"], errors="coerce").dropna().tail(120)
    if len(prices) < 10:
        return _smooth_fallback_forecast(current_price, base_date, horizon)

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    recent_returns = returns.tail(30)
    trend = float(recent_returns.mean()) if not recent_returns.empty else 0.0
    trend = float(np.clip(trend, -0.003, 0.003))
    volatility = float(recent_returns.std()) if len(recent_returns) > 1 else 0.01
    volatility = float(np.clip(volatility, 0.004, 0.035))

    forecast = []
    price = float(current_price)
    for day in range(1, horizon + 1):
        target_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        seasonal = 0.0015 * np.sin(day / 30 * np.pi)
        price = price * (1 + trend + seasonal)
        interval = volatility * np.sqrt(day / 7)
        forecast.append({
            "date": target_date,
            "min": float(price * max(0.0, 1 - interval)),
            "predicted": float(price),
            "max": float(price * (1 + interval)),
        })
    return forecast


# Load models globally to avoid reloading on every request
RISK_MODEL = load_pkl("risk_rf.pkl")
DURIAN_MODEL = load_pkl("xgb_sau_rieng_ri6.pkl")
OOLONG_MODEL = load_pkl("xgb_che_o_long.pkl")

import requests
import time

WEATHER_CACHE_TTL_SECONDS = int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "600"))
_WEATHER_CACHE = {}


def _first_value(values, default=None):
    if isinstance(values, list) and values:
        return values[0]
    return default


def _round_weather(value, default):
    if value is None:
        value = default
    return round(float(value), 1)


def get_weather(location_name=None):
    """Fetch current and daily weather from Open-Meteo."""
    now = time.time()
    cached = _WEATHER_CACHE.get(location_name)
    if cached and now - cached["fetched_at"] < WEATHER_CACHE_TTL_SECONDS:
        return cached["data"]

    loc_info = LOCATION_MAPPING.get(location_name) or next(iter(LOCATION_MAPPING.values()))
    lat = loc_info["coordinates"]["lat"]
    lon = loc_info["coordinates"]["lon"]

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,rain,showers,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "Asia/Ho_Chi_Minh",
                "forecast_days": 1,
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        daily_temp_max = _first_value(daily.get("temperature_2m_max"), 28.5)
        daily_temp_min = _first_value(daily.get("temperature_2m_min"), 19.0)
        daily_precipitation = _first_value(daily.get("precipitation_sum"), 0.0)
        current_temp = current.get("temperature_2m", daily_temp_max)
        current_precipitation = current.get("precipitation", 0.0)

        weather = {
            "temp_current": _round_weather(current_temp, daily_temp_max),
            "temp_max": _round_weather(daily_temp_max, current_temp),
            "temp_min": _round_weather(daily_temp_min, current_temp),
            "precipitation": _round_weather(daily_precipitation, 0.0),
            "precipitation_current": _round_weather(current_precipitation, 0.0),
            "humidity": current.get("relative_humidity_2m"),
            "weather_code": current.get("weather_code"),
            "weather_time": current.get("time"),
            "source": "open-meteo",
            "is_fallback": False,
        }
        _WEATHER_CACHE[location_name] = {"fetched_at": now, "data": weather}
        return weather
    except Exception as e:
        print(f"Error fetching weather: {e}")
        weather = {
            "temp_current": 28.5,
            "temp_max": 28.5,
            "temp_min": 19.0,
            "precipitation": 15.0,
            "precipitation_current": 0.0,
            "humidity": None,
            "weather_code": None,
            "weather_time": None,
            "source": "fallback",
            "is_fallback": True,
        }
        _WEATHER_CACHE[location_name] = {"fetched_at": now, "data": weather}
        return weather


def predict_risk(location_name, crop_name):
    """
    Dự báo mức độ rủi ro bằng Random Forest
    """
    if not RISK_MODEL:
        return 0, 0.1 # Default safe
    
    loc_info = LOCATION_MAPPING.get(location_name, {})
    elevation = loc_info.get("elevation", 1000)
    weather = get_weather(location_name)
    
    # Chuẩn bị input cho RF: ['temp_max', 'temp_min', 'precipitation', 'elevation']
    X = pd.DataFrame([[
        weather['temp_max'], 
        weather['temp_min'], 
        weather['precipitation'], 
        elevation
    ]], columns=['temp_max', 'temp_min', 'precipitation', 'elevation'])
    
    risk_level = int(RISK_MODEL.predict(X)[0])
    risk_proba = RISK_MODEL.predict_proba(X).max()
    
    return risk_level, risk_proba

def predict_price(crop_name, current_price, location_name="Phường B'Lao"):
    """
    Dự báo giá bằng XGBoost hoặc TFT
    """
    crop_info = CROP_MAPPING.get(crop_name, {})
    model_type = crop_info.get("model_type")
    
    # ── Chọn model tương ứng ──
    model = None
    if crop_name == "Sầu riêng Ri6":
        model = DURIAN_MODEL
    elif crop_name == "Chè Ô Long":
        model = OOLONG_MODEL
    elif crop_name in ["Cà phê Robusta", "Cà phê Arabica"]:
        # Placeholder cho TFT model (sẽ load từ pkl/pth sau khi train)
        # model = load_pkl(f"tft_{crop_name.lower().replace(' ', '_')}.pkl")
        pass
        
    forecast = []
    base_date = datetime.now()
    
    if model and model_type == "xgboost":
        # Dự báo đệ quy (Recursive multi-step forecast) cho 30 ngày
        temp_price = current_price
        weather = get_weather(location_name)
        
        # Lấy các đặc trưng thời gian
        current_date = base_date
        
        for i in range(1, 31):
            current_date += timedelta(days=1)
            month = current_date.month
            quarter = (month - 1) // 3 + 1
            day_of_week = current_date.weekday()
            
            # Features: ['month', 'quarter', 'day_of_week', 'temp_max', 'temp_min', 'precipitation', 'price_vnd_Lag_1', '7', '14', '30']
            # Dùng temp_price làm Lag_1, và giả lập các lag khác từ temp_price
            X = pd.DataFrame([[
                month, quarter, day_of_week,
                weather['temp_max'], weather['temp_min'], weather['precipitation'],
                temp_price, temp_price * 0.99, temp_price * 1.01, temp_price * 0.98
            ]], columns=['month', 'quarter', 'day_of_week', 'temp_max', 'temp_min', 'precipitation', 
                         'price_vnd_Lag_1', 'price_vnd_Lag_7', 'price_vnd_Lag_14', 'price_vnd_Lag_30'])
            
            pred_val = float(model.predict(X)[0])
            
            # Giới hạn biến động thực tế (không quá 2% mỗi ngày)
            max_change = temp_price * 0.02
            pred_val = max(temp_price - max_change, min(temp_price + max_change, pred_val))
            
            forecast.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "min": pred_val * 0.97,
                "predicted": pred_val,
                "max": pred_val * 1.03
            })
            temp_price = pred_val # Cập nhật cho ngày kế tiếp
            
    elif model_type == "tft":
        return _historical_price_forecast(crop_name, current_price, base_date)
    else:
        # Fallback
        for i in range(1, 31):
            target_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            seasonal_adjustment = 0.005 * np.sin(i / 30 * np.pi)
            predicted = current_price * (1 + seasonal_adjustment)
            forecast.append({
                "date": target_date,
                "min": float(predicted * 0.95),
                "predicted": float(predicted),
                "max": float(predicted * 1.05)
            })
            
    return forecast

def get_latest_price(crop_name):
    """
    Lấy giá mới nhất từ file CSV dữ liệu lịch sử.
    Nếu không tìm thấy, trả về giá mặc định an toàn.
    """
    try:
        file_base = CROP_FILE_MAP.get(crop_name)
        if not file_base:
            # Fallback nếu không khớp tên (cho sầu riêng, chè...)
            clean_name = crop_name.lower()
            if "sầu riêng" in clean_name: file_base = "durian_ri6"
            elif "chè" in clean_name or "ô long" in clean_name: file_base = "oolong"
            else: return 120000.0 # Giá mặc định chung

        file_name = f"processed_{file_base}.csv"
        csv_path = os.path.join(DATA_DIR, file_name)
        
        if os.path.exists(csv_path):
            # Chỉ đọc dòng cuối cùng để tiết kiệm RAM
            df = pd.read_csv(csv_path)
            if not df.empty:
                return float(df.iloc[-1]['price_vnd'])
    except Exception as e:
        print(f"⚠️ Lỗi khi lấy giá mới nhất cho {crop_name}: {e}")
    
    # Giá fallback dựa trên loại cây nếu có lỗi
    fallbacks = {
        "Cà phê Robusta": 120000.0,
        "Cà phê Arabica": 150000.0,
        "Sầu riêng Ri6": 115000.0,
        "Chè Ô Long": 250000.0
    }
    return fallbacks.get(crop_name, 100000.0)
