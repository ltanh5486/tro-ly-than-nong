"""
Registry hoạt chất/nhóm thuốc tham khảo cho module bệnh sầu riêng.

Nguyên tắc:
- Không thay thế nhãn thuốc và đăng ký sử dụng hiện hành.
- Không tự đưa liều lượng cố định.
- Không tự suy diễn thuốc nếu bệnh/triệu chứng chưa đủ chắc chắn.
- Ưu tiên IPM: vệ sinh vườn, tỉa tán, thoát nước, quản lý côn trùng.
"""

from __future__ import annotations

from typing import Any


REGULATORY_NOTE = (
    "Chỉ sử dụng sản phẩm có trong danh mục thuốc BVTV được phép sử dụng "
    "tại Việt Nam và có đăng ký phù hợp với cây sầu riêng/đối tượng gây hại. "
    "Đọc đúng nhãn, tuân thủ thời gian cách ly, trang bị bảo hộ và nguyên tắc 4 đúng."
)


PESTICIDE_REGISTRY: dict[str, dict[str, Any]] = {

    "Leaf_Algal": {
        "vi_name": "Đốm tảo trên lá",
        "recommendation_level": "cần xác nhận",
        "groups": [
            {
                "group": "hợp chất đồng",
                "active_ingredients": [
                    "copper hydroxide",
                    "copper oxychloride",
                ],
                "note": (
                    "Chỉ cân nhắc khi đã xác nhận đốm tảo/nhiễm bề mặt "
                    "và sản phẩm có đăng ký phù hợp."
                ),
            }
        ],
    },

    "Leaf_Blight": {
        "vi_name": "Cháy lá",
        "recommendation_level": "không kê thuốc chỉ từ ảnh",
        "groups": [],
        "note": (
            "Cháy lá là triệu chứng có thể do nhiều tác nhân. "
            "Cần xác định thêm nguyên nhân trước khi chọn hoạt chất."
        ),
    },

    "Leaf_Colletotrichum": {
        "vi_name": "Bệnh lá do Colletotrichum",
        "recommendation_level": "tham khảo sau xác nhận",
        "groups": [
            {
                "group": "thuốc nấm tiếp xúc",
                "active_ingredients": [
                    "mancozeb",
                    "copper hydroxide",
                ],
            },
            {
                "group": "thuốc nấm nội hấp",
                "active_ingredients": [
                    "azoxystrobin",
                    "difenoconazole",
                    "tebuconazole",
                ],
            },
        ],
    },

    "Leaf_Healthy": {
        "vi_name": "Lá khỏe",
        "recommendation_level": "không sử dụng thuốc",
        "groups": [],
        "note": "Không khuyến cáo sử dụng thuốc BVTV khi cây đang khỏe.",
    },

    "Leaf_Phomopsis": {
        "vi_name": "Bệnh lá do Phomopsis",
        "recommendation_level": "tham khảo sau xác nhận",
        "groups": [
            {
                "group": "thuốc nấm phổ rộng",
                "active_ingredients": [
                    "mancozeb",
                    "copper hydroxide",
                ],
            },
            {
                "group": "thuốc nấm nội hấp",
                "active_ingredients": [
                    "azoxystrobin",
                    "difenoconazole",
                ],
            },
        ],
    },

    "Leaf_Rhizoctonia": {
        "vi_name": "Bệnh do Rhizoctonia",
        "recommendation_level": "tham khảo sau xác nhận",
        "groups": [
            {
                "group": "thuốc nấm",
                "active_ingredients": [
                    "validamycin",
                    "azoxystrobin",
                ],
            }
        ],
    },

    "anthracnose_disease": {
        "vi_name": "Bệnh thán thư",
        "recommendation_level": "tham khảo sau xác nhận",
        "groups": [
            {
                "group": "thuốc nấm tiếp xúc",
                "active_ingredients": [
                    "mancozeb",
                    "copper hydroxide",
                ],
            },
            {
                "group": "thuốc nấm nội hấp",
                "active_ingredients": [
                    "azoxystrobin",
                    "difenoconazole",
                    "tebuconazole",
                ],
            },
        ],
    },

    "canker_disease": {
        "vi_name": "Bệnh loét thân/cành",
        "recommendation_level": "cần xác định tác nhân",
        "groups": [],
        "note": (
            "Không khuyến cáo hoạt chất cố định vì loét thân/cành "
            "có thể do nhiều tác nhân khác nhau."
        ),
    },

    "fruit_rot": {
        "vi_name": "Thối trái",
        "recommendation_level": "phân biệt tác nhân trước",
        "groups": [
            {
                "group": "nhóm dùng khi nghi nấm thật",
                "active_ingredients": [
                    "azoxystrobin",
                    "difenoconazole",
                    "mancozeb",
                ],
            },
            {
                "group": "nhóm dùng khi xác nhận Phytophthora/oomycete",
                "active_ingredients": [
                    "fosetyl-aluminium",
                    "phosphorous acid / phosphonate",
                    "metalaxyl-M",
                ],
            },
        ],
        "note": (
            "Phải phân biệt thối trái do nấm thật với nhóm Phytophthora "
            "trước khi lựa chọn thuốc."
        ),
    },

    "mealybug_infestation": {
        "vi_name": "Rệp sáp",
        "recommendation_level": "ưu tiên IPM",
        "groups": [
            {
                "group": "dầu/khoáng hoặc chế phẩm tiếp xúc",
                "active_ingredients": [
                    "mineral oil",
                ],
            },
            {
                "group": "thuốc trừ côn trùng",
                "active_ingredients": [
                    "spirotetramat",
                    "pymetrozine",
                    "dinotefuran",
                ],
            },
        ],
        "note": (
            "Quản lý kiến và ổ rệp trước; ưu tiên sinh học/IPM. "
            "Chỉ dùng thuốc khi mật số cần kiểm soát."
        ),
    },

    "pink_disease": {
        "vi_name": "Bệnh nấm hồng",
        "recommendation_level": "tham khảo sau xác nhận",
        "groups": [
            {
                "group": "thuốc nấm",
                "active_ingredients": [
                    "copper hydroxide",
                    "copper oxychloride",
                    "hexaconazole",
                ],
            }
        ],
    },

    "sooty_mold": {
        "vi_name": "Nấm bồ hóng",
        "recommendation_level": "xử lý nguyên nhân chính",
        "groups": [],
        "note": (
            "Không ưu tiên thuốc nấm. Cần kiểm soát rệp sáp/rệp chích hút "
            "và kiến là nguồn dịch ngọt nuôi nấm bồ hóng."
        ),
    },

    "stem_blight": {
        "vi_name": "Cháy thân/cành",
        "recommendation_level": "cần xác định tác nhân",
        "groups": [],
        "note": (
            "Cắt bỏ mô bệnh, vệ sinh dụng cụ và xác định tác nhân "
            "trước khi chọn thuốc."
        ),
    },

    "stem_cracking_ gummosis": {
        "vi_name": "Nứt thân, xì mủ",
        "recommendation_level": "ưu tiên kiểm tra Phytophthora",
        "groups": [
            {
                "group": "nhóm phosphonate/phosphite",
                "active_ingredients": [
                    "fosetyl-aluminium",
                    "phosphorous acid / phosphonate",
                ],
            },
            {
                "group": "nhóm phenylamide",
                "active_ingredients": [
                    "metalaxyl-M",
                ],
            },
        ],
        "note": (
            "Chỉ áp dụng nhóm này khi triệu chứng và kiểm tra thực địa "
            "phù hợp với bệnh do Phytophthora/oomycete."
        ),
    },

    "thrips_disease": {
        "vi_name": "Bọ trĩ gây hại",
        "recommendation_level": "ưu tiên IPM và luân phiên nhóm",
        "groups": [
            {
                "group": "thuốc trừ bọ trĩ",
                "active_ingredients": [
                    "spinetoram",
                    "spinosad",
                    "abamectin",
                ],
            }
        ],
        "note": (
            "Luân phiên nhóm tác động để hạn chế kháng thuốc; "
            "chỉ xử lý khi mật số đạt mức cần can thiệp."
        ),
    },

    "yellow_leaf": {
        "vi_name": "Vàng lá",
        "recommendation_level": "không kê thuốc chỉ từ triệu chứng",
        "groups": [],
        "note": (
            "Vàng lá có thể do dinh dưỡng, úng, rễ, pH hoặc sâu bệnh. "
            "Phải xác định nguyên nhân trước khi sử dụng thuốc BVTV."
        ),
    },
}


def get_pesticide_recommendation(class_name: str) -> dict[str, Any]:
    """
    Trả về registry thuốc/hoạt chất tham khảo theo class của model.
    """

    item = PESTICIDE_REGISTRY.get(class_name)

    if item is None:
        return {
            "class_name": class_name,
            "vi_name": class_name,
            "recommendation_level": "chưa có dữ liệu",
            "groups": [],
            "note": "Chưa có khuyến nghị thuốc cho lớp này.",
            "regulatory_note": REGULATORY_NOTE,
        }

    return {
        "class_name": class_name,
        **item,
        "regulatory_note": REGULATORY_NOTE,
    }