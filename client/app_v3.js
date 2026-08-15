/**
 * 🌾 Trợ Lý Thần Nông - Frontend Logic (app.js)
 * Xử lý: Gửi form, Fetch API, Render UI, Chart.js
 */

// ── Cấu hình ─────────────────────────────────────────────────
const CONFIG = {
    // Tự động sử dụng localhost nếu đang chạy máy ảo, hoặc dùng domain hiện tại nếu deploy chung
    API_BASE_URL: window.location.origin + "/api",
    SIMULATED_DELAY: 2000, 
};

// ── State Management ─────────────────────────────────────────
let state = {
    user: null, // {email, full_name, role, token}
    charts: {
        weather: null,
        price: null,
        finance: null
    }
};

// ── Khởi tạo ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    initEventListeners();
    updateCapitalValue();
    initAuthListeners(); // Mới: Lắng nghe sự kiện đăng nhập/đăng ký

    // Tự động mở modal nếu có tham số ?auth=login
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('auth') === 'login' && !state.user) {
        const authModal = document.getElementById('auth-modal');
        if (authModal) authModal.classList.add('active');
    }
});

// ── Event Listeners ──────────────────────────────────────────
function initEventListeners() {
    const form = document.getElementById('predict-form');
    const capitalSlider = document.getElementById('capital-slider');
    const errorClose = document.getElementById('error-close');
    const floatingBtn = document.getElementById('floating-btn');

    // Submit Form
    form.addEventListener('submit', handleFormSubmit);

    // Live update slider value
    capitalSlider.addEventListener('input', updateCapitalValue);

    // Đóng banner lỗi
    errorClose.addEventListener('click', () => {
        document.getElementById('error-banner').classList.add('hidden');
    });


    // Mobile: Floating button cuộn lên đầu
    floatingBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Theo dõi cuộn để hiện/ẩn floating button
    window.addEventListener('scroll', handleScroll);
}

// ── Xử lý Form ───────────────────────────────────────────────
async function handleFormSubmit(e) {
    e.preventDefault();

    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    
    // Thu thập dữ liệu
    const location = document.getElementById('location-select').value;
    const crop = document.getElementById('crop-select').value;

    // Validation bổ sung cho test TC015 (nhìn thấy text trong DOM)
    if (!location || !crop) {
        const testMsg = document.getElementById('test-validation-msg');
        if (testMsg) {
            testMsg.classList.remove('hidden');
            testMsg.style.display = 'block';
        }
        showError("Vui lòng chọn vùng trồng và cây trồng.");
        return;
    } else {
        const testMsg = document.getElementById('test-validation-msg');
        if (testMsg) {
            testMsg.classList.add('hidden');
            testMsg.style.display = 'none';
        }
    }

    const formData = {
        location: location,
        crop: crop,
        mode: document.querySelector('input[name="farming-mode"]:checked').value,
        capital: parseFloat(document.getElementById('capital-slider').value),
        area_ha: parseFloat(document.getElementById('area-input').value)
    };

    // UI: Disable button & hiện spinner
    submitBtn.disabled = true;
    btnText.classList.add('hidden');
    btnSpinner.classList.remove('hidden');
    hideError();

    // Hiệu ứng xoay vòng thông điệp (Cải tiến số 3)
    const loadingMsgs = [
        "🔍 Đang quét 6 năm dữ liệu...",
        "🌤️ Đang lấy thời tiết Real-time...",
        "🤖 AI đang dự báo giá (TFT)...",
        "💰 Đang tính toán ROI & Vốn...",
        "✅ Đang chuẩn bị khuyến nghị..."
    ];
    let msgIdx = 0;
    const statusEl = btnSpinner.querySelector('span');
    const msgInterval = setInterval(() => {
        msgIdx = (msgIdx + 1) % loadingMsgs.length;
        statusEl.innerHTML = `<span class="status-rotate">${loadingMsgs[msgIdx]}</span>`;
    }, 1500);

    try {
        // Gỡ bỏ giả lập delay để tối ưu tốc độ phản hồi thực tế
        // await new Promise(resolve => setTimeout(resolve, 3500));

        // Fetch API
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${state.user?.token}`
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `Lỗi server (${response.status})`);
        }

        // Thành công: Render kết quả
        renderResults(data);

    } catch (error) {
        console.error("API Error:", error);
        showError(error.message);
    } finally {
        // Hoàn tất: Reset button
        clearInterval(msgInterval);
        submitBtn.disabled = false;
        btnText.classList.remove('hidden');
        btnSpinner.classList.add('hidden');
        statusEl.textContent = "Đang xử lý...";
    }
}

// ── Render UI ────────────────────────────────────────────────
function renderResults(data) {
    // Xóa class is-empty để form trượt sang trái, hiện dashboard
    document.getElementById('app-container').classList.remove('is-empty');
    document.getElementById('results-container').classList.remove('hidden');

    // 1. Thông tin vùng
    renderLocationInfo(data.location_info);

    // 2. Hộp Khuyến Nghị
    renderRecommendation(data);

    // 3. Biểu đồ (Sẽ hoàn thiện ở Bước 12)
    renderCharts(data);
}

function renderLocationInfo(info) {
    const container = document.getElementById('location-details');
    
    // Xác định icon thời tiết dựa trên lượng mưa và nhiệt độ
    const tempIcon = info.current_temp >= 30 ? '🔥' : info.current_temp >= 22 ? '☀️' : '🌤️';
    const rainIcon = info.recent_rainfall_mm >= 100 ? '🌧️' : info.recent_rainfall_mm >= 20 ? '🌦️' : '💧';
    const elevIcon = info.elevation >= 1200 ? '🏔️' : '⛰️';
    
    const todayStr = new Date().toLocaleDateString('vi-VN');
    const bLaoDesc = "Đặc điểm: Vùng đất đỏ bazan màu mỡ, địa hình đồi dốc nhẹ. Khí hậu mát mẻ quanh năm, mưa nhiều và sương mù bao phủ, rất thích hợp cho cây công nghiệp lâu năm đặc biệt là trà và cà phê.";
    const xuanTruongDesc = "Đặc điểm: Nằm ở độ cao lớn, sương mù nhiều, biên độ nhiệt ngày đêm lớn. Đất đai tơi xốp, thích hợp cho cây trồng ôn đới và đặc sản như chè Ô Long.";
    const desc = info.name === "Phường B'Lao" ? bLaoDesc : xuanTruongDesc;

    // Thêm tỉnh Lâm Đồng vào tên hiển thị
    const locationDisplayName = info.name === "Phường B'Lao" 
        ? "Phường B'Lao, tỉnh Lâm Đồng" 
        : "Phường Xuân Trường, tỉnh Lâm Đồng";

    container.innerHTML = `
        <div class="location-header-info">
            <span class="location-name">${locationDisplayName}</span>
            <span class="location-climate text-sm text-gray-500">Cập nhật: ${todayStr}</span>
            <p class="location-climate mt-1">${info.climate || 'Đang cập nhật'}</p>
        </div>
        <div class="location-stat-grid">
            <div class="stat-item stat-temp">
                <span class="stat-icon">${tempIcon}</span>
                <div class="stat-data">
                    <span class="stat-value font-mono">${info.current_temp}°C</span>
                    <span class="stat-label">Nhiệt độ hiện tại</span>
                </div>
            </div>
            <div class="stat-item stat-rain">
                <span class="stat-icon">${rainIcon}</span>
                <div class="stat-data">
                    <span class="stat-value font-mono">${info.recent_rainfall_mm}mm</span>
                    <span class="stat-label">Mưa hiện tại</span>
                </div>
            </div>
            <div class="stat-item stat-elevation">
                <span class="stat-icon">${elevIcon}</span>
                <div class="stat-data">
                    <span class="stat-value font-mono">${info.elevation}m</span>
                    <span class="stat-label">Độ cao</span>
                </div>
            </div>
        </div>
        <div class="location-description mt-4 p-3 bg-white bg-opacity-50 rounded-lg text-sm text-gray-700">
            ${desc}
        </div>
    `;
}

function renderRecommendation(data) {
    const box = document.getElementById('recommendation-box');
    const header = document.getElementById('rec-header');
    const icon = document.getElementById('rec-icon');
    const level = document.getElementById('rec-level');
    const message = document.getElementById('rec-message');
    const trend = document.getElementById('rec-trend');
    const decision = document.getElementById('rec-decision');
    const reasoning = document.getElementById('reasoning-text');
    const alternatives = document.getElementById('rec-alternatives');
    const checklistContainer = document.getElementById('rec-checklist');

    const plan = data.action_plan;
    
    // Reset classes
    box.className = 'card recommendation-box';
    
    // Màu sắc theo level
    if (plan.level === 'danger') {
        box.classList.add('bg-red-50', 'border-red-200');
        icon.textContent = '🔴';
        level.textContent = 'MỨC ĐỘ: TỪ CHỐI / NGUY HIỂM';
        level.className = 'rec-level text-red-800';
    } else if (plan.level === 'warning') {
        box.classList.add('bg-yellow-50', 'border-yellow-200');
        icon.textContent = '🟡';
        level.textContent = 'MỨC ĐỘ: THẬN TRỌNG';
        level.className = 'rec-level text-yellow-800';
    } else {
        box.classList.add('bg-green-50', 'border-green-200');
        icon.textContent = '🟢';
        level.textContent = 'MỨC ĐỘ: AN TOÀN';
        level.className = 'rec-level text-green-800';
    }

    message.textContent = plan.message;
    reasoning.textContent = plan.reasoning || "Không có giải thích chi tiết.";

    // Cảnh báo giá vật tư
    const fertilizerBox = document.getElementById('rec-fertilizer');
    const fertilizerText = document.getElementById('fertilizer-text');
    if (data.fertilizer_advice) {
        fertilizerText.textContent = data.fertilizer_advice;
        fertilizerBox.classList.remove('hidden');
    } else {
        fertilizerBox.classList.add('hidden');
    }

    // Dự báo giá & Quyết định (chỉ hiện khi không phải danger)
    if (plan.level !== 'danger' && data.price_forecast) {
        const pf = data.price_forecast;
        const trendIcon = pf.trend === 'up' ? '📈' : '📉';
        const trendText = pf.trend === 'up' ? 'Tăng' : 'Giảm';
        trend.innerHTML = `${trendIcon} Giá dự báo: <span class="font-bold">${trendText} ${pf.trend_pct.toFixed(1)}%</span> trong 30 ngày tới.`;
        decision.textContent = `📢 Khuyến nghị: ${data.production_decision.recommendation}`;
        decision.classList.remove('hidden');
        trend.classList.remove('hidden');
        checklistContainer.classList.remove('hidden');
        alternatives.classList.add('hidden');
        
        renderChecklist(plan.checklist);
    } else {
        // Case Danger: Hiện gợi ý thay thế
        trend.classList.add('hidden');
        decision.classList.add('hidden');
        checklistContainer.classList.add('hidden');
        renderAlternatives(data.crop_alternatives);
    }
}

function renderChecklist(items) {
    const list = document.getElementById('checklist-list');
    list.innerHTML = '';
    
    if (!items || items.length === 0) {
        list.innerHTML = '<li class="text-sm text-muted">Không có hành động khẩn cấp.</li>';
        return;
    }

    items.forEach((item, index) => {
        const li = document.createElement('li');
        li.className = 'checklist-item flex items-start gap-2 mb-1';
        
        const bullet = document.createElement('span');
        bullet.className = 'text-primary text-sm';
        bullet.textContent = '•';
        
        const textSpan = document.createElement('span');
        textSpan.className = 'text-sm';
        textSpan.textContent = item.action;

        li.appendChild(bullet);
        li.appendChild(textSpan);
        list.appendChild(li);
    });
}

function renderAlternatives(crops) {
    const container = document.getElementById('rec-alternatives');
    const tags = document.getElementById('alternatives-tags');
    tags.innerHTML = '';

    if (!crops || crops.length === 0) {
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');
    crops.forEach(crop => {
        const span = document.createElement('span');
        span.className = 'alt-tag bg-white border border-red-200 text-red-700 px-3 py-1 rounded-full text-xs cursor-pointer hover:bg-red-700 hover:text-white transition-colors';
        span.textContent = crop;
        span.onclick = () => selectCropAndAnalyze(crop);
        tags.appendChild(span);
    });
}

function selectCropAndAnalyze(crop) {
    const select = document.getElementById('crop-select');
    select.value = crop;
    document.getElementById('predict-form').dispatchEvent(new Event('submit'));
}

// ── Helpers ──────────────────────────────────────────────────
function updateCapitalValue() {
    const slider = document.getElementById('capital-slider');
    const display = document.getElementById('capital-value');
    const unitDisplay = document.querySelector('#capital-display .slider-unit');
    const val = parseFloat(slider.value);
    
    // Update text
    if (val >= 1000000000) {
        display.textContent = (val / 1000000000).toFixed(1);
        if (unitDisplay) unitDisplay.textContent = 'tỷ VND';
    } else {
        display.textContent = (val / 1000000).toFixed(0);
        if (unitDisplay) unitDisplay.textContent = 'triệu VND';
    }

    // Dynamic track coloring for "Premium" look
    const min = slider.min || 0;
    const max = slider.max || 100;
    const percentage = (val - min) / (max - min) * 100;
    slider.style.backgroundSize = percentage + '% 100%';
}

function showError(msg) {
    const banner = document.getElementById('error-banner');
    const message = document.getElementById('error-message');
    message.textContent = msg;
    banner.classList.remove('hidden');
    // Tự động cuộn tới lỗi
    banner.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    document.getElementById('error-banner').classList.add('hidden');
}

function handleScroll() {
    const btn = document.getElementById('floating-btn');
    if (window.scrollY > 500) {
        btn.classList.remove('hidden');
    } else {
        btn.classList.add('hidden');
    }
}


// ── Chart Logic (Bước 12) ─────────────────────
function renderCharts(data) {
    // 1. Weather Chart (Luôn hiện nếu có location_info)
    if (data.location_info) {
        renderWeatherChart(data.location_info);
        document.getElementById('weather-overlay').classList.add('hidden');
    } else {
        document.getElementById('weather-overlay').classList.remove('hidden');
    }

    const isDanger = data.action_plan && data.action_plan.level === 'danger';
    const isKienThiet = data.farming_mode === 'Kiến thiết';

    // 2. Price Forecast Chart (Ẩn nếu Risk Level 2 hoặc Kiến thiết)
    const priceOverlay = document.getElementById('price-overlay');
    if (data.price_forecast && !isDanger && !isKienThiet) {
        renderPriceChart(data.price_forecast);
        priceOverlay.classList.add('hidden');
    } else {
        let priceText = "📊 Dữ liệu đang được cập nhật";
        if (isDanger) {
            priceText = "⚠️ Cảnh báo rủi ro sinh thái cao.<br><span class='text-sm font-normal mt-1 block'>Dự báo giá không khả dụng do không thể canh tác.</span>";
        } else if (isKienThiet) {
            priceText = "⚠️ Khuyến cáo: Cây trồng cần 3-5 năm mới cho thu hoạch.<br><span class='text-sm font-normal mt-1 block'>Dự báo giá ngắn hạn không áp dụng cho giai đoạn này.</span>";
        }

        priceOverlay.querySelector('span').innerHTML = priceText;
        if (isDanger) priceOverlay.classList.add('danger-overlay');
        else priceOverlay.classList.remove('danger-overlay');
        priceOverlay.classList.remove('hidden');
    }

    // 3. Financial Chart (Ẩn nếu Risk Level 2 hoặc data rỗng)
    const financeOverlay = document.getElementById('finance-overlay');
    if (data.financial_analysis && !isDanger) {
        renderFinanceChart(data.financial_analysis);
        financeOverlay.classList.add('hidden');
    } else {
        let financeText = "📊 Dữ liệu đang được cập nhật";
        if (isDanger) {
            financeText = "⚠️ Cảnh báo rủi ro sinh thái cao.<br><span class='text-sm font-normal mt-1 block'>Phân tích tài chính không khả dụng do không thể canh tác.</span>";
        }
        
        financeOverlay.querySelector('span').innerHTML = financeText;
        if (isDanger) financeOverlay.classList.add('danger-overlay');
        else financeOverlay.classList.remove('danger-overlay');
        financeOverlay.classList.remove('hidden');
    }
}

function renderWeatherChart(info) {
    const ctx = document.getElementById('weather-chart').getContext('2d');
    
    if (state.charts.weather) state.charts.weather.destroy();

    // Tạo labels ngày thực tế (dd/MM) thay vì "Ngày 1, Ngày 2..."
    const today = new Date();
    const labels = Array.from({length: 14}, (_, i) => {
        const d = new Date(today);
        d.setDate(d.getDate() + i);
        return `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`;
    });
    const tempMax = labels.map((_, i) => i === 0 ? Number(info.current_temp || 0) : null);
    const rainfall = labels.map((_, i) => i === 0 ? Number(info.recent_rainfall_mm || 0) : null);

    state.charts.weather = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Lượng mưa (mm)',
                    data: rainfall,
                    backgroundColor: 'rgba(82, 183, 136, 0.5)',
                    borderColor: '#52B788',
                    borderWidth: 1,
                    yAxisID: 'yRain'
                },
                {
                    label: 'Nhiệt độ (°C)',
                    data: tempMax,
                    type: 'line',
                    borderColor: '#F4A261',
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    tension: 0.4,
                    yAxisID: 'yTemp',
                    pointBackgroundColor: '#F4A261',
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                yRain: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Lượng mưa (mm)', font: { weight: 'bold' } },
                    beginAtZero: true
                },
                yTemp: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Nhiệt độ (°C)', font: { weight: 'bold' } },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function renderPriceChart(pf) {
    const ctx = document.getElementById('price-chart').getContext('2d');
    
    if (state.charts.price) state.charts.price.destroy();

    // Format ngày dd/MM từ YYYY-MM-DD
    const labels = pf.forecast_30_days.map(p => {
        const parts = p.date.split('-');
        return `${parts[2]}/${parts[1]}`;
    });
    const predicted = pf.forecast_30_days.map(p => p.predicted);
    const mins = pf.forecast_30_days.map(p => p.min);
    const maxs = pf.forecast_30_days.map(p => p.max);

    // Lấy đơn vị từ API (VND/kg mặc định)
    const priceUnit = pf.unit || 'VNĐ/kg';

    state.charts.price = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Giá dự báo',
                    data: predicted,
                    borderColor: '#F4A261',
                    backgroundColor: 'rgba(244, 162, 97, 0.1)',
                    fill: false,
                    tension: 0.3,
                    borderWidth: 3,
                    pointRadius: 2,
                    pointBackgroundColor: '#F4A261'
                },
                {
                    label: 'Biên độ Tin cậy (Max)',
                    data: maxs,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(82, 183, 136, 0.15)',
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: 'Biên độ Tin cậy (Min)',
                    data: mins,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(82, 183, 136, 0.15)',
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: priceUnit,
                        font: { weight: 'bold', size: 12 },
                        padding: { bottom: 15 }
                    },
                    ticks: {
                        callback: function(value) {
                            return new Intl.NumberFormat('vi-VN').format(value);
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('vi-VN').format(Math.round(context.parsed.y)) + ' ' + priceUnit;
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

function renderFinanceChart(fa) {
    const ctx = document.getElementById('finance-chart').getContext('2d');
    
    if (state.charts.finance) state.charts.finance.destroy();

    // Lấy cost breakdown từ API (hoặc fallback)
    const cb = fa.cost_breakdown || {
        seeds: fa.estimated_cost * 0.25,
        fertilizer: fa.estimated_cost * 0.45,
        labor: fa.estimated_cost * 0.30
    };

    // Auto format: nếu giá trị > 1 tỷ → hiển thị tỷ, ngược lại hiển thị triệu
    const maxVal = Math.max(fa.estimated_cost, fa.estimated_revenue);
    const useBillion = maxVal >= 1_000_000_000;
    const divisor = useBillion ? 1_000_000_000 : 1_000_000;
    const unitLabel = useBillion ? 'tỷ VNĐ' : 'triệu VNĐ';

    const currentMode = document.querySelector('input[name="farming-mode"]:checked')?.value || "Kinh doanh";
    const seedLabel = currentMode === "Kiến thiết" ? '🏗️ Cơ sở hạ tầng & Vật tư' : '📦 Khấu hao & Vật tư khác';
    const costLabel = currentMode === "Kiến thiết" ? 'Chi phí Đầu tư (Năm 1)' : 'Tổng Chi Phí (Năm)';

    state.charts.finance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [costLabel, 'Doanh Thu Kỳ Vọng'],
            datasets: [
                {
                    label: seedLabel,
                    data: [cb.seeds, 0],
                    backgroundColor: '#F4A261',
                    stack: 'finance'
                },
                {
                    label: '🧪 Phân bón',
                    data: [cb.fertilizer, 0],
                    backgroundColor: '#E76F51',
                    stack: 'finance'
                },
                {
                    label: '👷 Nhân công',
                    data: [cb.labor, 0],
                    backgroundColor: '#D4A373',
                    stack: 'finance'
                },
                {
                    label: '💰 Doanh thu',
                    data: [0, fa.estimated_revenue],
                    backgroundColor: fa.estimated_profit >= 0 ? '#2D6A4F' : '#DC2626',
                    stack: 'finance'
                }
            ]
        },
        plugins: [{
            id: 'topLabels',
            afterDatasetsDraw(chart, args, pluginOptions) {
                const { ctx, data } = chart;
                ctx.save();
                ctx.font = 'bold 13px "Be Vietnam Pro"';
                ctx.fillStyle = '#1e293b'; // Tailwind slate-800
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';

                // Lấy bar đầu tiên (stack cost) và bar cuối cùng (stack revenue)
                const costBarMeta = chart.getDatasetMeta(2); // Nhân công (top của stack 0)
                const revBarMeta = chart.getDatasetMeta(3); // Doanh thu (top của stack 1)
                
                const costBar = costBarMeta.data[0];
                const revBar = revBarMeta.data[1];

                const formatLabel = (val) => {
                    if (val >= 1_000_000_000) {
                        return (val / 1_000_000_000).toFixed(1) + ' tỷ VNĐ';
                    }
                    return (val / 1_000_000).toFixed(0) + ' triệu VNĐ';
                };

                if (costBar && fa.estimated_cost > 0) {
                    ctx.fillText(formatLabel(fa.estimated_cost), costBar.x, costBar.y - 5);
                }

                if (revBar && fa.estimated_revenue > 0) {
                    ctx.fillText(formatLabel(fa.estimated_revenue), revBar.x, revBar.y - 5);
                }

                ctx.restore();
            }
        }],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { font: { size: 11 }, usePointStyle: true, pointStyle: 'rectRounded' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (context.parsed.y !== null && context.parsed.y !== 0) {
                                const val = context.parsed.y / 1_000_000;
                                label += ': ' + val.toFixed(1) + ' triệu VNĐ';
                            }
                            return context.parsed.y !== 0 ? label : null;
                        },
                        afterBody: function(tooltipItems) {
                            if (tooltipItems[0] && tooltipItems[0].dataIndex === 1) {
                                const profit = fa.estimated_profit / 1_000_000;
                                const profitSign = profit >= 0 ? '+' : '';
                                return [`\n📊 Lợi nhuận: ${profitSign}${profit.toFixed(1)} triệu VNĐ`, `📈 ROI: ${fa.roi_pct.toFixed(1)}%`];
                            }
                            return [];
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { display: false }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    grace: '20%',
                    title: {
                        display: true,
                        text: unitLabel,
                        font: { weight: 'bold', size: 12 }
                    },
                    ticks: {
                        callback: function(value) {
                            if (value >= 1_000_000_000) {
                                return (value / 1_000_000_000).toFixed(1) + ' tỷ';
                            }
                            return (value / 1_000_000).toFixed(0) + ' tr';
                        }
                    }
                }
            }
        }
    });
}

// ── Chat Logic (Agentic RAG) ─────────────────────
let chatHistory = [];

function initChat() {
    const chatWidget = document.getElementById('chat-widget');
    const chatToggle = document.getElementById('chat-toggle');
    const chatClose = document.getElementById('chat-close');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    if (!chatWidget || !chatToggle) return;

    // Mở/Đóng chat
    chatToggle.addEventListener('click', () => {
        chatWidget.classList.toggle('closed');
        // Hiện gợi ý khi mở chat nếu chưa có tin nhắn nào mới
        if (chatHistory.length === 0) {
            renderChatSuggestions([
                "Giá sầu riêng hôm nay?",
                "Kỹ thuật bón phân cà phê",
                "Phòng bệnh rỉ sắt",
                "Cách tăng năng suất chè"
            ]);
        }
    });
    chatClose.addEventListener('click', () => chatWidget.classList.add('closed'));

    // Gửi tin nhắn
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        // 1. Add user message
        addMessage('user', text);
        chatInput.value = '';
        
        // 2. Add typing indicator
        const typingId = addTypingIndicator();
        
        try {
            // Đã loại bỏ giả lập suy nghĩ để tăng tốc độ phản hồi

            // Lấy ngữ cảnh hiện tại từ form chính
            const context = {
                location: document.getElementById('location-select').value,
                crop: document.getElementById('crop-select').value,
                mode: document.querySelector('input[name="farming-mode"]:checked')?.value || "Kinh doanh",
                capital: parseFloat(document.getElementById('capital-slider').value),
                area_ha: parseFloat(document.getElementById('area-input').value)
            };

            const response = await fetch(`${CONFIG.API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${state.user?.token}`
                },
                body: JSON.stringify({
                    message: text,
                    history: chatHistory,
                    context: context
                })
            });

            const data = await response.json();
            removeTypingIndicator(typingId);

            if (data.status === 'success' || data.answer) {
                addMessage('assistant', data.answer);
                if (data.suggestions) renderChatSuggestions(data.suggestions);
                
                // Cập nhật lịch sử
                chatHistory.push({ role: 'user', content: text });
                chatHistory.push({ role: 'assistant', content: data.answer });
                if (chatHistory.length > 10) chatHistory.splice(0, 2); 
            } else {
                addMessage('assistant', '⚠️ Xin lỗi bà con, tôi đang bị gián đoạn kết nối. Bà con thử lại sau nhé!');
            }
        } catch (error) {
            removeTypingIndicator(typingId);
            console.error("Chat Error:", error);
            addMessage('assistant', '❌ Lỗi kết nối máy chủ AI.');
        }
    });
}

function renderChatSuggestions(suggestions) {
    const container = document.getElementById('chat-suggestions');
    if (!container) return;
    container.innerHTML = '';
    
    // Chỉ lấy tối đa 4 gợi ý cho gọn
    suggestions.slice(0, 4).forEach(text => {
        const chip = document.createElement('div');
        chip.className = 'suggestion-chip';
        chip.textContent = text;
        chip.onclick = () => {
            const input = document.getElementById('chat-input');
            input.value = text;
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
            // Xóa gợi ý sau khi chọn
            container.innerHTML = '';
        };
        container.appendChild(chip);
    });
}

function addMessage(role, text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    let formattedText = text;

    if (role === 'assistant') {
        // Format bold text: **text** → <strong>text</strong>
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Format italic text: _text_ → <em>text</em>
        formattedText = formattedText.replace(/(?<!\w)_(.*?)_(?!\w)/g, '<em>$1</em>');
        
        // Format bullet points: -, *, • text → <li>
        formattedText = formattedText.replace(
            /^[ \t]*[-*•]\s+(.+)$/gm,
            '<li style="margin-left: 20px; list-style-type: disc; margin-bottom: 4px; line-height: 1.5;">$1</li>'
        );
        
        // Gom các thẻ <li> lại thành <ul>
        formattedText = formattedText.replace(/(<li[^>]*>.*?<\/li>\n?)+/g, function(match) {
            return `<ul style="margin-top: 8px; margin-bottom: 8px; padding-left: 10px;">\n${match}</ul>\n`;
        });
        
        // Format xuống dòng
        formattedText = formattedText.replace(/\n/g, '<br>');
        
        // Xóa các <br> thừa do <ul> và <li> tạo ra
        formattedText = formattedText.replace(/<br><ul/g, '<ul');
        formattedText = formattedText.replace(/<\/ul><br>/g, '</ul>');
        formattedText = formattedText.replace(/<\/li><br>/g, '</li>');
    } else {
        // User message: chỉ cần xuống dòng
        formattedText = formattedText.replace(/\n/g, '<br>');
    }
    
    div.innerHTML = formattedText;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ── Auth Logic ───────────────────────────────────────────────

function closeChangePasswordModal() {
    const modal = document.getElementById('change-password-modal');
    if (modal) {
        modal.classList.add('hidden');
        // Reset form
        const form = document.getElementById('change-password-form');
        if (form) form.reset();
        const errorMsg = document.getElementById('change-password-error');
        if (errorMsg) errorMsg.classList.add('hidden');
    }
}

function initAuthListeners() {
    const tabLogin = document.getElementById('tab-login');
    const tabRegister = document.getElementById('tab-register');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const logoutBtn = document.getElementById('logout-btn');

    tabLogin.addEventListener('click', () => {
        tabLogin.classList.add('active');
        tabRegister.classList.remove('active');
        loginForm.classList.remove('hidden');
        registerForm.classList.add('hidden');
    });

    tabRegister.addEventListener('click', () => {
        tabRegister.classList.add('active');
        tabLogin.classList.remove('active');
        registerForm.classList.remove('hidden');
        loginForm.classList.add('hidden');
    });

    loginForm.addEventListener('submit', handleLogin);
    registerForm.addEventListener('submit', handleRegister);
    logoutBtn.addEventListener('click', logout);

    // Mới: Sự kiện đổi mật khẩu
    document.getElementById('change-pwd-btn')?.addEventListener('click', openChangePasswordModal);
    document.getElementById('change-password-form')?.addEventListener('submit', handleChangePassword);

    // Forgot Password Flow
    const showForgotBtn = document.getElementById('show-forgot-btn');
    const backToLoginBtns = document.querySelectorAll('.back-to-login');
    const forgotForm = document.getElementById('forgot-password-form');
    const resetForm = document.getElementById('reset-password-form');

    showForgotBtn?.addEventListener('click', () => {
        loginForm.classList.add('hidden');
        registerForm.classList.add('hidden');
        forgotForm.classList.remove('hidden');
        document.getElementById('auth-error').classList.add('hidden');
    });

    backToLoginBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            forgotForm.classList.add('hidden');
            resetForm.classList.add('hidden');
            registerForm.classList.add('hidden');
            loginForm.classList.remove('hidden');
            document.getElementById('auth-error').classList.add('hidden');
        });
    });

    forgotForm?.addEventListener('submit', handleForgotPassword);
    resetForm?.addEventListener('submit', handleResetPassword);
}

async function handleForgotPassword(e) {
    e.preventDefault();
    const email = document.getElementById('forgot-email').value;
    const errorMsg = document.getElementById('auth-error');
    const forgotForm = document.getElementById('forgot-password-form');
    const resetForm = document.getElementById('reset-password-form');

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/auth/forgot-password`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Không thể gửi yêu cầu");

        alert(data.message);
        document.getElementById('reset-email-display').value = email;
        forgotForm.classList.add('hidden');
        resetForm.classList.remove('hidden');
        errorMsg.classList.add('hidden');
    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.remove('hidden');
    }
}

// ── Đổi mật khẩu (Cho người đang đăng nhập) ───────────────────
function openChangePasswordModal() {
    document.getElementById('change-password-modal').classList.remove('hidden');
    document.getElementById('user-dropdown').classList.add('hidden');
}

function closeChangePasswordModal() {
    document.getElementById('change-password-modal').classList.add('hidden');
    document.getElementById('change-password-form').reset();
    document.getElementById('change-password-error').classList.add('hidden');
}

async function handleChangePassword(e) {
    e.preventDefault();
    const oldPassword = document.getElementById('change-old-password').value;
    const newPassword = document.getElementById('change-new-password').value;
    const confirmPassword = document.getElementById('change-confirm-password').value;
    const errorMsg = document.getElementById('change-password-error');

    if (newPassword !== confirmPassword) {
        errorMsg.textContent = "Mật khẩu mới không khớp!";
        errorMsg.classList.remove('hidden');
        return;
    }

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/auth/change-password`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${state.user.token}`
            },
            body: JSON.stringify({
                old_password: oldPassword,
                new_password: newPassword
            })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Đổi mật khẩu thất bại");

        alert("✅ Đổi mật khẩu thành công!");
        closeChangePasswordModal();
    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.remove('hidden');
    }
}

async function handleResetPassword(e) {
    e.preventDefault();
    const email = document.getElementById('reset-email-display').value;
    const otp = document.getElementById('reset-otp').value;
    const newPassword = document.getElementById('reset-password').value;
    const errorMsg = document.getElementById('auth-error');

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/auth/reset-password`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: email,
                otp: otp,
                new_password: newPassword
            })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Đặt lại mật khẩu thất bại");

        alert(data.message);
        location.reload(); // Quay lại trang login mặc định
    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.remove('hidden');
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const errorMsg = document.getElementById('auth-error');

    try {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        const response = await fetch(`${CONFIG.API_BASE_URL}/auth/login`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Đăng nhập thất bại");

        loginUser(data);
    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.remove('hidden');
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const name = document.getElementById('reg-name').value;
    const username = document.getElementById('reg-username').value;
    const email = document.getElementById('reg-email').value;
    const password = document.getElementById('reg-password').value;
    const errorMsg = document.getElementById('auth-error');

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username: username,
                email: email,
                password: password,
                full_name: name,
                role: "farmer"
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Đăng ký thất bại");

        // Sau khi đăng ký thành công, tự động đăng nhập hoặc chuyển sang tab login
        alert("Đăng ký thành công! Mời bà con đăng nhập.");
        document.getElementById('tab-login').click();
    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.remove('hidden');
    }
}

function loginUser(data) {
    const user = {
        token: data.access_token,
        role: data.role,
        full_name: data.full_name || "Người dùng"
    };
    localStorage.setItem('user', JSON.stringify(user));
    state.user = user;
    applyAuthState();
}

function checkAuth() {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
        state.user = JSON.parse(savedUser);
        applyAuthState();
    }
}

function applyAuthState() {
    const authModal = document.getElementById('auth-modal');
    const userHeader = document.getElementById('user-header');
    const displayUserName = document.getElementById('display-user-name');
    const displayUserRole = document.getElementById('display-user-role');
    const adminLink = document.getElementById('admin-link');

    if (state.user) {
        authModal.classList.add('hidden');
        userHeader.classList.remove('hidden');
        
        // Hiển thị nút Admin nếu có quyền
        if (state.user.role === 'admin') {
            adminLink.classList.remove('hidden');
        } else {
            adminLink.classList.add('hidden');
        }
        
        displayUserName.textContent = state.user.full_name;
        displayUserRole.textContent = state.user.role === 'admin' ? 'Quản trị viên' : 'Nông dân';

        // Cập nhật Avatar chữ cái
        const firstLetter = state.user.full_name ? state.user.full_name.charAt(0).toUpperCase() : 'U';
        const headerLetter = document.getElementById('header-avatar-letter');
        const dropdownLetter = document.getElementById('dropdown-avatar-letter');
        if (headerLetter) headerLetter.textContent = firstLetter;
        if (dropdownLetter) dropdownLetter.textContent = firstLetter;
        
        if (state.user.role === 'admin') {
            displayUserRole.classList.add('bg-orange-600');
        }

        // Tải lại lịch sử chat
        loadChatHistory();
    } else {
        authModal.classList.remove('hidden');
        userHeader.classList.add('hidden');
    }
}

async function loadChatHistory() {
    if (!state.user || !state.user.token) return;
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/chat/history`, {
            headers: { 'Authorization': `Bearer ${state.user.token}` }
        });
        const history = await response.json();
        
        const chatContainer = document.getElementById('chat-messages');
        chatContainer.innerHTML = ''; // Xóa tin nhắn mặc định
        
        history.forEach(item => {
            addMessage('user', item.question);
            addMessage('assistant', item.answer);
        });
        
        // Cuộn xuống cuối
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } catch (error) {
        console.error('Lỗi tải lịch sử chat:', error);
    }
}

function logout() {
    localStorage.removeItem('user');
    state.user = null;
    location.reload(); // Reset toàn bộ state
}

function addTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant typing-indicator';
    div.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function renderChatSuggestions(suggestions) {
    const container = document.getElementById('chat-suggestions');
    if (!container) return;
    container.innerHTML = '';
    
    suggestions.forEach(s => {
        const chip = document.createElement('div');
        chip.className = 'suggestion-chip';
        chip.textContent = s;
        chip.onclick = () => {
            document.getElementById('chat-input').value = s;
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
        };
        container.appendChild(chip);
    });
}

// Khởi tạo Chat khi load trang
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    initAuthListeners();
    initChat();
    initUserMenu();
});

function initUserMenu() {
    const toggle = document.getElementById('user-menu-toggle');
    const dropdown = document.getElementById('user-dropdown');

    if (!toggle || !dropdown) return;

    toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdown.classList.toggle('hidden');
    });

    document.addEventListener('click', () => {
        dropdown.classList.add('hidden');
    });

    dropdown.addEventListener('click', (e) => {
        e.stopPropagation();
    });
}

// ═══════════════════════════════════════════════════════════════
// 🌿 NHẬN DIỆN BỆNH SẦU RIÊNG
// ═══════════════════════════════════════════════════════════════

const DISEASE_NAMES_VI = {
    "Leaf_Algal": "Đốm rong trên lá",
    "Leaf_Blight": "Cháy lá",
    "Leaf_Colletotrichum": "Bệnh Colletotrichum trên lá",
    "Leaf_Healthy": "Lá khỏe mạnh",
    "Leaf_Phomopsis": "Bệnh Phomopsis trên lá",
    "Leaf_Rhizoctonia": "Bệnh Rhizoctonia trên lá",
    "anthracnose_disease": "Bệnh thán thư",
    "canker_disease": "Bệnh loét thân/cành",
    "fruit_rot": "Thối trái",
    "mealybug_infestation": "Rệp sáp",
    "pink_disease": "Bệnh nấm hồng",
    "sooty_mold": "Nấm bồ hóng",
    "stem_blight": "Cháy thân",
    "stem_cracking_ gummosis": "Nứt thân, chảy nhựa",
    "thrips_disease": "Bọ trĩ",
    "yellow_leaf": "Vàng lá"
};

function initDiseaseDetection() {
    const imageInput = document.getElementById('disease-image');
    const predictBtn = document.getElementById('disease-predict-btn');
    const preview = document.getElementById('disease-preview');
    const previewWrapper = document.getElementById('disease-preview-wrapper');
    const fileName = document.getElementById('disease-file-name');

    if (!imageInput || !predictBtn) return;

    imageInput.addEventListener('change', () => {
        resetDiseaseResult();

        const file = imageInput.files?.[0];

        if (!file) {
            predictBtn.disabled = true;
            if (fileName) fileName.textContent = 'Chưa chọn ảnh';
            previewWrapper?.classList.add('hidden');
            return;
        }

        const allowedTypes = [
            'image/jpeg',
            'image/png',
            'image/webp'
        ];

        if (!allowedTypes.includes(file.type)) {
            showDiseaseError(
                'Chỉ hỗ trợ ảnh JPG, JPEG, PNG hoặc WEBP.'
            );
            imageInput.value = '';
            predictBtn.disabled = true;
            if (fileName) fileName.textContent = 'Chưa chọn ảnh';
            previewWrapper?.classList.add('hidden');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showDiseaseError(
                'Ảnh vượt quá dung lượng tối đa 10 MB.'
            );
            imageInput.value = '';
            predictBtn.disabled = true;
            if (fileName) fileName.textContent = 'Chưa chọn ảnh';
            previewWrapper?.classList.add('hidden');
            return;
        }

        if (fileName) {
            fileName.textContent = file.name;
        }

        predictBtn.disabled = false;

        const objectUrl = URL.createObjectURL(file);

        if (preview) {
            preview.src = objectUrl;
            preview.onload = () => {
                URL.revokeObjectURL(objectUrl);
            };
        }

        previewWrapper?.classList.remove('hidden');
    });

    predictBtn.addEventListener(
        'click',
        handleDiseasePrediction
    );
}


async function handleDiseasePrediction() {
    const imageInput = document.getElementById('disease-image');
    const predictBtn = document.getElementById('disease-predict-btn');
    const loading = document.getElementById('disease-loading');
    const resultBox = document.getElementById('disease-result');
    const errorBox = document.getElementById('disease-error');

    const file = imageInput?.files?.[0];

    if (!file) {
        showDiseaseError(
            'Vui lòng chọn ảnh sầu riêng trước.'
        );
        return;
    }

    predictBtn.disabled = true;
    loading?.classList.remove('hidden');
    resultBox?.classList.add('hidden');
    errorBox?.classList.add('hidden');

    const formData = new FormData();

    formData.append('image', file);
    formData.append('plant_part', 'leaf');
    formData.append('symptoms', '');

    const selectedLocation =
        document.getElementById('location-select')?.value || '';

    formData.append('location', selectedLocation);

    try {
        const headers = {};

        if (state.user?.token) {
            headers['Authorization'] =
                `Bearer ${state.user.token}`;
        }

        const response = await fetch(
            `${CONFIG.API_BASE_URL}/disease/predict`,
            {
                method: 'POST',
                headers,
                body: formData
            }
        );

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                `Máy chủ trả về dữ liệu không hợp lệ (${response.status}).`
            );
        }

        if (!response.ok) {
            let message = '';

            if (Array.isArray(data.detail)) {
                message = data.detail
                    .map(item => item.msg || JSON.stringify(item))
                    .join('; ');
            } else {
                message =
                    data.detail ||
                    `Lỗi máy chủ (${response.status})`;
            }

            throw new Error(message);
        }

        if (!data.prediction) {
            throw new Error(
                'Máy chủ không trả về kết quả nhận diện.'
            );
        }

        renderDiseaseResult(data.prediction);

    } catch (error) {
        console.error(
            'Disease Detection Error:',
            error
        );

        showDiseaseError(
            error.message ||
            'Không thể phân tích ảnh. Vui lòng thử lại.'
        );

    } finally {
        loading?.classList.add('hidden');

        if (imageInput?.files?.[0]) {
            predictBtn.disabled = false;
        }
    }
}


function renderDiseaseResult(prediction) {
    const resultBox =
        document.getElementById('disease-result');

    const diseaseName =
        document.getElementById('disease-name');

    const confidence =
        document.getElementById('disease-confidence');

    const top3 =
        document.getElementById('disease-top3');

    const errorBox =
        document.getElementById('disease-error');

    errorBox?.classList.add('hidden');

    const rawDisease =
        prediction.disease || '';

    const viName =
        DISEASE_NAMES_VI[rawDisease] ||
        rawDisease ||
        'Chưa xác định';

    if (diseaseName) {
        diseaseName.textContent = viName;
    }

    const mainConfidence =
        Number(prediction.confidence || 0);

    if (confidence) {
        confidence.textContent =
            `${(mainConfidence * 100).toFixed(1)}%`;
    }

    if (top3) {
        top3.innerHTML = '';
    }

    const predictions =
        Array.isArray(prediction.top_predictions)
            ? prediction.top_predictions
            : [];

    predictions.forEach((item, index) => {
        const row =
            document.createElement('div');

        row.className =
            'disease-top-item';

        const itemConfidence =
            Number(item.confidence || 0);

        const percent =
            Math.max(
                0,
                Math.min(
                    100,
                    itemConfidence * 100
                )
            );

        const name =
            DISEASE_NAMES_VI[item.class] ||
            item.class ||
            'Không xác định';

        row.innerHTML = `
            <span class="disease-top-name">
                ${index + 1}. ${escapeDiseaseHtml(name)}
            </span>

            <div class="disease-top-bar">
                <div
                    class="disease-top-bar-fill"
                    style="width: ${percent.toFixed(1)}%"
                ></div>
            </div>

            <span class="disease-top-confidence">
                ${percent.toFixed(1)}%
            </span>
        `;

        top3?.appendChild(row);
    });

    resultBox?.classList.remove('hidden');

    resultBox?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
    });
}


function showDiseaseError(message) {
    const errorBox =
        document.getElementById('disease-error');

    const resultBox =
        document.getElementById('disease-result');

    const loading =
        document.getElementById('disease-loading');

    resultBox?.classList.add('hidden');
    loading?.classList.add('hidden');

    if (errorBox) {
        errorBox.textContent =
            `⚠️ ${message}`;
        errorBox.classList.remove('hidden');
    }
}


function resetDiseaseResult() {
    const resultBox =
        document.getElementById('disease-result');

    const errorBox =
        document.getElementById('disease-error');

    resultBox?.classList.add('hidden');
    errorBox?.classList.add('hidden');
}


function escapeDiseaseHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}


// Khởi tạo nhận diện bệnh khi trang load xong
document.addEventListener(
    'DOMContentLoaded',
    initDiseaseDetection
);