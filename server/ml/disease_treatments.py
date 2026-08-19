"""
Knowledge base hỗ trợ xử lý bệnh sầu riêng.

File này độc lập với mô hình ONNX.
Không thực hiện suy luận ảnh và không thay đổi disease_inference.py.

Nguyên tắc:
- AI nhận diện bệnh -> disease_inference.py
- Kiến thức xử lý -> disease_treatments.py
- API kết hợp hai kết quả ở bước sau.
"""

from __future__ import annotations

from typing import Any


COMMON_WARNING = (
    "Khuyến nghị chỉ mang tính hỗ trợ. Khi sử dụng thuốc bảo vệ thực vật, "
    "chỉ sử dụng sản phẩm đang được phép lưu hành và có hướng dẫn phù hợp "
    "với cây sầu riêng/đối tượng gây hại; tuân thủ đúng nhãn, liều lượng, "
    "thời gian cách ly và nguyên tắc 4 đúng. Không tự ý phối trộn thuốc."
)


DISEASE_TREATMENTS: dict[str, dict[str, Any]] = {

    # ============================================================
    # 1. TẢO ĐỎ / ĐỐM TẢO
    # ============================================================
    "Leaf_Algal": {
        "vi_name": "Đốm tảo trên lá",
        "category": "bệnh lá",
        "likely_cause": "Tảo ký sinh trên bề mặt lá, thường phát triển mạnh khi vườn ẩm và tán cây rậm.",
        "symptoms": [
            "Xuất hiện các đốm tròn hoặc gần tròn trên lá.",
            "Đốm có thể chuyển màu vàng cam, nâu hoặc đỏ gạch.",
            "Bệnh thường nặng hơn trong điều kiện ẩm độ cao.",
        ],
        "management": [
            "Tỉa cành để vườn thông thoáng.",
            "Giảm ẩm độ kéo dài trong tán cây.",
            "Thu gom lá bị bệnh nặng.",
            "Cân đối dinh dưỡng, tránh để cây suy yếu.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Chưa tự động đề xuất hoạt chất khi chỉ dựa trên ảnh. "
            "Cần xác nhận tác nhân và đối chiếu danh mục thuốc hiện hành."
        ),
    },

    # ============================================================
    # 2. CHÁY LÁ
    # ============================================================
    "Leaf_Blight": {
        "vi_name": "Cháy lá",
        "category": "bệnh lá",
        "likely_cause": (
            "Triệu chứng cháy lá có thể liên quan đến nhiều tác nhân nấm "
            "hoặc điều kiện bất lợi; cần kết hợp triệu chứng thực địa."
        ),
        "symptoms": [
            "Mép hoặc đầu lá chuyển nâu và khô.",
            "Vết bệnh có thể lan rộng làm cháy từng mảng lá.",
            "Trường hợp nặng có thể gây rụng lá.",
        ],
        "management": [
            "Cắt bỏ lá và cành bị bệnh nặng.",
            "Thu gom tàn dư bệnh ra khỏi vườn.",
            "Tạo tán thông thoáng và hạn chế lá ẩm kéo dài.",
            "Kiểm tra hệ thống tưới và thoát nước.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Không nên chọn thuốc chỉ dựa vào nhãn 'cháy lá'. "
            "Cần xác định thêm tác nhân trước khi chọn hoạt chất."
        ),
    },

    # ============================================================
    # 3. COLLETOTRICHUM
    # ============================================================
    "Leaf_Colletotrichum": {
        "vi_name": "Bệnh lá do Colletotrichum",
        "category": "bệnh nấm",
        "likely_cause": "Nấm thuộc chi Colletotrichum.",
        "symptoms": [
            "Đốm nâu hoặc nâu đen trên lá.",
            "Vết bệnh có thể mở rộng và liên kết thành mảng.",
            "Điều kiện nóng ẩm thuận lợi cho bệnh phát triển.",
        ],
        "management": [
            "Loại bỏ bộ phận bị bệnh nặng.",
            "Tỉa tán và giảm ẩm độ trong vườn.",
            "Hạn chế làm tổn thương mô cây.",
            "Theo dõi cả lá, cành, hoa và trái.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Nếu xác nhận Colletotrichum, hệ thống thuốc ở bước tiếp theo "
            "sẽ tra cứu hoạt chất được phép sử dụng hiện hành."
        ),
    },

    # ============================================================
    # 4. LÁ KHỎE
    # ============================================================
    "Leaf_Healthy": {
        "vi_name": "Lá khỏe",
        "category": "không phát hiện bệnh",
        "likely_cause": None,
        "symptoms": [
            "AI chưa phát hiện dấu hiệu bệnh nổi bật thuộc các lớp đã học."
        ],
        "management": [
            "Tiếp tục theo dõi cây định kỳ.",
            "Duy trì dinh dưỡng cân đối.",
            "Giữ vườn thông thoáng và thoát nước tốt.",
        ],
        "active_ingredients": [],
        "chemical_note": "Không khuyến cáo sử dụng thuốc bảo vệ thực vật.",
    },

    # ============================================================
    # 5. PHOMOPSIS
    # ============================================================
    "Leaf_Phomopsis": {
        "vi_name": "Bệnh lá do Phomopsis",
        "category": "bệnh nấm",
        "likely_cause": "Nấm thuộc nhóm Phomopsis/Diaporthe.",
        "symptoms": [
            "Xuất hiện vết hoại tử trên lá.",
            "Vết bệnh có thể lan rộng khi độ ẩm cao.",
            "Có thể liên quan đến hiện tượng khô cành ở cây suy yếu.",
        ],
        "management": [
            "Cắt bỏ mô bệnh và vệ sinh dụng cụ cắt.",
            "Giảm độ ẩm trong tán.",
            "Không để vườn úng nước.",
            "Tăng cường sức khỏe cây bằng dinh dưỡng cân đối.",
        ],
        "active_ingredients": [],
        "chemical_note": "Cần xác nhận tác nhân trước khi lựa chọn thuốc.",
    },

    # ============================================================
    # 6. RHIZOCTONIA
    # ============================================================
    "Leaf_Rhizoctonia": {
        "vi_name": "Bệnh do Rhizoctonia",
        "category": "bệnh nấm",
        "likely_cause": "Nấm Rhizoctonia spp.",
        "symptoms": [
            "Mô lá xuất hiện vùng cháy hoặc hoại tử.",
            "Bệnh thường thuận lợi trong điều kiện ẩm cao.",
        ],
        "management": [
            "Giữ vườn thông thoáng.",
            "Hạn chế nước đọng và ẩm kéo dài.",
            "Loại bỏ bộ phận bệnh nặng.",
            "Kiểm tra đồng thời vùng cổ rễ và đất quanh cây.",
        ],
        "active_ingredients": [],
        "chemical_note": "Cần xác nhận vị trí và mức độ bệnh trước khi xử lý hóa học.",
    },

    # ============================================================
    # 7. THÁN THƯ
    # ============================================================
    "anthracnose_disease": {
        "vi_name": "Bệnh thán thư",
        "category": "bệnh nấm",
        "likely_cause": "Thường liên quan đến nấm Colletotrichum spp.",
        "symptoms": [
            "Vết bệnh màu nâu đến nâu đen.",
            "Có thể xuất hiện trên lá, cành, hoa hoặc trái.",
            "Bệnh phát triển mạnh trong điều kiện mưa ẩm.",
        ],
        "management": [
            "Cắt và tiêu hủy bộ phận bệnh nặng.",
            "Tỉa tán để giảm ẩm.",
            "Hạn chế tưới làm ướt tán cây kéo dài.",
            "Theo dõi bệnh sát trong mùa mưa.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Hoạt chất sẽ được tra từ lớp dữ liệu thuốc BVTV hiện hành "
            "trước khi hiển thị cho người dùng."
        ),
    },

    # ============================================================
    # 8. LOÉT / CANKER
    # ============================================================
    "canker_disease": {
        "vi_name": "Bệnh loét thân/cành",
        "category": "bệnh thân cành",
        "likely_cause": (
            "Có thể do nhiều tác nhân. Ảnh đơn lẻ không đủ để khẳng định "
            "nguyên nhân gây loét."
        ),
        "symptoms": [
            "Vỏ thân hoặc cành xuất hiện vùng tổn thương.",
            "Mô bệnh có thể đổi màu, nứt hoặc khô.",
        ],
        "management": [
            "Kiểm tra toàn bộ thân và cành.",
            "Vệ sinh dụng cụ trước và sau khi cắt.",
            "Loại bỏ mô chết khi phù hợp.",
            "Hạn chế gây vết thương cơ giới trên cây.",
        ],
        "active_ingredients": [],
        "chemical_note": "Cần xác định tác nhân trước khi khuyến nghị thuốc.",
    },

    # ============================================================
    # 9. THỐI TRÁI
    # ============================================================
    "fruit_rot": {
        "vi_name": "Thối trái",
        "category": "bệnh trái",
        "likely_cause": (
            "Có thể liên quan đến nấm hoặc oomycete; cần kiểm tra thêm "
            "vết bệnh, cuống trái và điều kiện vườn."
        ),
        "symptoms": [
            "Trái xuất hiện vùng nâu, mềm hoặc thối.",
            "Vết bệnh có thể lan nhanh khi thời tiết ẩm.",
        ],
        "management": [
            "Thu gom và loại bỏ trái bệnh.",
            "Không để trái bệnh tồn tại trong vườn.",
            "Giữ tán thông thoáng.",
            "Kiểm soát nước và thoát nước tốt trong mùa mưa.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Cần phân biệt nhóm tác nhân trước khi chọn thuốc vì thuốc "
            "đối với nấm thật và oomycete không hoàn toàn giống nhau."
        ),
    },

    # ============================================================
    # 10. RỆP SÁP
    # ============================================================
    "mealybug_infestation": {
        "vi_name": "Rệp sáp",
        "category": "côn trùng gây hại",
        "likely_cause": "Rệp sáp chích hút trên cây.",
        "symptoms": [
            "Xuất hiện cụm côn trùng trắng dạng bột/sáp.",
            "Cây có thể suy yếu do bị chích hút.",
            "Dịch ngọt có thể tạo điều kiện cho nấm bồ hóng.",
        ],
        "management": [
            "Kiểm tra mật số rệp trên lá, cành và trái.",
            "Cắt bỏ bộ phận bị nhiễm nặng khi cần.",
            "Quản lý kiến vì kiến có thể bảo vệ và phát tán rệp.",
            "Ưu tiên biện pháp sinh học và bảo tồn thiên địch.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Chỉ đề xuất thuốc trừ côn trùng sau khi đối chiếu sản phẩm "
            "được đăng ký cho đối tượng/cây trồng phù hợp."
        ),
    },

    # ============================================================
    # 11. NẤM HỒNG
    # ============================================================
    "pink_disease": {
        "vi_name": "Bệnh nấm hồng",
        "category": "bệnh nấm thân cành",
        "likely_cause": "Nấm gây bệnh nấm hồng trên thân/cành.",
        "symptoms": [
            "Cành hoặc thân xuất hiện lớp nấm màu hồng.",
            "Cành bệnh có thể suy yếu và khô.",
        ],
        "management": [
            "Cắt bỏ cành bệnh nặng.",
            "Tiêu hủy tàn dư bệnh.",
            "Tỉa tán thông thoáng.",
            "Theo dõi các cành nằm trong vùng ẩm và thiếu ánh sáng.",
        ],
        "active_ingredients": [],
        "chemical_note": "Chỉ sử dụng thuốc sau khi đối chiếu đăng ký hiện hành.",
    },

    # ============================================================
    # 12. NẤM BỒ HÓNG
    # ============================================================
    "sooty_mold": {
        "vi_name": "Nấm bồ hóng",
        "category": "bệnh thứ cấp",
        "likely_cause": (
            "Nấm bồ hóng thường phát triển trên dịch ngọt do côn trùng "
            "chích hút như rệp tiết ra."
        ),
        "symptoms": [
            "Bề mặt lá hoặc cành phủ lớp màu đen như muội.",
            "Có thể làm giảm khả năng quang hợp.",
        ],
        "management": [
            "Tìm và kiểm soát nguồn côn trùng chích hút.",
            "Quản lý rệp và kiến.",
            "Vệ sinh bề mặt cây khi phù hợp.",
            "Tạo tán thông thoáng.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Ưu tiên xử lý nguyên nhân là côn trùng tiết dịch ngọt thay "
            "vì chỉ phun thuốc trị lớp nấm bề mặt."
        ),
    },

    # ============================================================
    # 13. CHÁY THÂN/CÀNH
    # ============================================================
    "stem_blight": {
        "vi_name": "Cháy thân/cành",
        "category": "bệnh thân cành",
        "likely_cause": "Có thể do nhiều tác nhân nấm gây bệnh thân cành.",
        "symptoms": [
            "Thân hoặc cành chuyển nâu và khô.",
            "Phần phía trên vùng bệnh có thể suy yếu.",
        ],
        "management": [
            "Cắt bỏ cành chết hoặc bệnh nặng.",
            "Khử trùng dụng cụ cắt.",
            "Bảo vệ vết cắt.",
            "Kiểm tra điều kiện thoát nước và sức khỏe bộ rễ.",
        ],
        "active_ingredients": [],
        "chemical_note": "Cần xác định tác nhân trước khi chọn thuốc.",
    },

    # ============================================================
    # 14. NỨT THÂN / XÌ MỦ
    # ============================================================
    "stem_cracking_ gummosis": {
        "vi_name": "Nứt thân, xì mủ",
        "category": "bệnh thân/rễ",
        "likely_cause": (
            "Triệu chứng có thể liên quan đến bệnh hại thân/rễ, trong đó "
            "cần đặc biệt kiểm tra nhóm tác nhân gây xì mủ và điều kiện úng."
        ),
        "symptoms": [
            "Vỏ thân nứt.",
            "Có dịch hoặc nhựa chảy từ vùng tổn thương.",
            "Mô quanh vết bệnh có thể chuyển màu.",
        ],
        "management": [
            "Kiểm tra cổ rễ và hệ thống thoát nước.",
            "Không để nước đọng quanh gốc.",
            "Loại bỏ mô chết theo hướng dẫn kỹ thuật.",
            "Hạn chế làm tổn thương thân.",
            "Theo dõi sự lan rộng của vết bệnh.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Không tự động kê thuốc chỉ từ ảnh. Cần xác nhận tác nhân, "
            "đặc biệt khi nghi bệnh liên quan đến Phytophthora."
        ),
    },

    # ============================================================
    # 15. BỌ TRĨ
    # ============================================================
    "thrips_disease": {
        "vi_name": "Bọ trĩ gây hại",
        "category": "côn trùng gây hại",
        "likely_cause": "Bọ trĩ chích hút mô non của cây.",
        "symptoms": [
            "Lá non có thể biến dạng hoặc bạc màu.",
            "Hoa và trái non có thể bị tổn thương.",
            "Mức độ hại tăng khi mật số bọ trĩ cao.",
        ],
        "management": [
            "Theo dõi mật số trên đọt non, hoa và trái non.",
            "Giữ vườn thông thoáng.",
            "Bảo tồn thiên địch.",
            "Chỉ xử lý hóa học khi mật số cần kiểm soát.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Khi cần dùng thuốc phải luân phiên nhóm tác động để hạn chế "
            "kháng thuốc và chỉ chọn sản phẩm có đăng ký phù hợp."
        ),
    },

    # ============================================================
    # 16. VÀNG LÁ
    # ============================================================
    "yellow_leaf": {
        "vi_name": "Vàng lá",
        "category": "triệu chứng tổng hợp",
        "likely_cause": (
            "Vàng lá không phải một nguyên nhân duy nhất. Có thể liên quan "
            "đến dinh dưỡng, bộ rễ, úng nước, sâu bệnh hoặc stress môi trường."
        ),
        "symptoms": [
            "Phiến lá chuyển vàng một phần hoặc toàn bộ.",
            "Cây có thể sinh trưởng yếu.",
        ],
        "management": [
            "Kiểm tra độ ẩm và thoát nước.",
            "Kiểm tra bộ rễ.",
            "Đánh giá dinh dưỡng và pH đất.",
            "Kiểm tra sâu bệnh đi kèm trước khi dùng thuốc.",
        ],
        "active_ingredients": [],
        "chemical_note": (
            "Không khuyến cáo thuốc chỉ dựa trên triệu chứng vàng lá. "
            "Phải xác định nguyên nhân trước."
        ),
    },
}


def get_disease_treatment(class_name: str) -> dict[str, Any]:
    """
    Lấy kiến thức xử lý tương ứng với class của model.

    Không phát sinh thuốc nếu class không tồn tại.
    """

    treatment = DISEASE_TREATMENTS.get(class_name)

    if treatment is None:
        return {
            "class_name": class_name,
            "vi_name": class_name,
            "category": "chưa xác định",
            "likely_cause": None,
            "symptoms": [],
            "management": [
                "Cần kiểm tra thêm triệu chứng thực địa trước khi xử lý."
            ],
            "active_ingredients": [],
            "chemical_note": (
                "Chưa có dữ liệu khuyến nghị cho lớp bệnh này."
            ),
            "warning": COMMON_WARNING,
        }

    return {
        "class_name": class_name,
        **treatment,
        "warning": COMMON_WARNING,
    }