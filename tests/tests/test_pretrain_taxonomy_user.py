from __future__ import annotations

from ml.tooling.core.pretrain_taxonomy_user import infer_user_taxonomy_label


def test_juvenile_justice_text_is_not_mislabeled_as_nsfw() -> None:
    text = (
        "未成年人司法社会支持体系用于刑事案件行为矫治和再社会化，也能服务犯罪预防。"
        "这一制度对未进入司法程序的未成年人权益保护具有重要价值。法院和律师应当依照"
        "法律提供帮助，并由监护人参与教育。"
    )

    assert infer_user_taxonomy_label(text) == "legal_statute"


def test_engineering_checks_are_not_mislabeled_as_medical() -> None:
    text = (
        "机械工程测试技术通过模型试验和现场实测评价设备性能，检查电路、载荷和应力。"
        "实验结果用于控制系统设计，最后列出设备安装的注意事项。"
    )

    assert infer_user_taxonomy_label(text) == "stem_engineering"


def test_clinical_guideline_remains_medical() -> None:
    text = (
        "肥厚型心肌病临床诊疗指南总结患者症状、诊断标准、治疗方案和药物不良反应，"
        "并给出适应证、禁忌证与循证推荐意见。"
    )

    assert infer_user_taxonomy_label(text) == "medical_guideline"


def test_bank_risk_management_is_finance_not_legal_or_medical() -> None:
    text = (
        "城市商业银行风险管理需要分析贷款、存款、净息差和资本充足率，"
        "并将经营策略与风控目标结合，以降低不良贷款。"
    )

    assert infer_user_taxonomy_label(text) == "finance_banking"


def test_explicit_pornographic_content_remains_restricted() -> None:
    assert infer_user_taxonomy_label("成人视频与色情影片下载页面") == "nsfw"


def test_ambiguous_substrings_do_not_trigger_nsfw() -> None:
    text = "道路交叉口交通进入建设高潮，黄文山在例题中用 xxx 表示未知数。"

    assert infer_user_taxonomy_label(text) != "nsfw"


def test_sexual_assault_statute_remains_legal() -> None:
    text = "《中华人民共和国刑法》规定，法院审理强奸犯罪案件时应当依法保护被害人。"

    assert infer_user_taxonomy_label(text) == "legal_statute"
