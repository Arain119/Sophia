from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic


SUITE_SCHEMA = "sophia_generation_quality_suite_v2"
REPORT_SCHEMA = "sophia_generation_quality_suite_build_v1"
EXPECTED_CASES = 2_000


_FACTS: tuple[tuple[str, str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    ("fact", "中国的首都是：", ("北京",), "The capital of China is:", ("Beijing",)),
    ("fact", "法国的首都是：", ("巴黎",), "The capital of France is:", ("Paris",)),
    ("fact", "日本的首都是：", ("东京",), "The capital of Japan is:", ("Tokyo",)),
    ("fact", "德国的首都是：", ("柏林",), "The capital of Germany is:", ("Berlin",)),
    ("fact", "意大利的首都是：", ("罗马",), "The capital of Italy is:", ("Rome",)),
    ("fact", "加拿大的首都是：", ("渥太华",), "The capital of Canada is:", ("Ottawa",)),
    ("fact", "澳大利亚的首都是：", ("堪培拉",), "The capital of Australia is:", ("Canberra",)),
    ("fact", "巴西的首都是：", ("巴西利亚",), "The capital of Brazil is:", ("Brasilia", "Brasília")),
    ("fact", "埃及的首都是：", ("开罗",), "The capital of Egypt is:", ("Cairo",)),
    ("fact", "印度的首都是：", ("新德里",), "The capital of India is:", ("New Delhi",)),
    ("fact", "世界上面积最大的海洋是：", ("太平洋",), "The largest ocean on Earth is the:", ("Pacific", "Pacific Ocean")),
    ("fact", "世界上海拔最高的山峰是：", ("珠穆朗玛峰",), "The highest mountain above sea level is:", ("Mount Everest", "Everest")),
    ("fact", "《红楼梦》的作者是：", ("曹雪芹",), "The author of Dream of the Red Chamber is:", ("Cao Xueqin",)),
    ("fact", "《哈姆雷特》的作者是：", ("莎士比亚",), "The author of Hamlet is:", ("William Shakespeare", "Shakespeare")),
    ("fact", "《蒙娜丽莎》的画家是：", ("达·芬奇", "达芬奇"), "The painter of the Mona Lisa was:", ("Leonardo da Vinci", "da Vinci")),
    ("fact", "巴西的官方语言是：", ("葡萄牙语",), "The primary language of Brazil is:", ("Portuguese",)),
    ("fact", "日本的货币是：", ("日元",), "The currency of Japan is the:", ("yen", "Japanese yen")),
    ("fact", "撒哈拉沙漠所在的大洲是：", ("非洲",), "The Sahara Desert is located in:", ("Africa",)),
    ("fact", "联合国总部所在的城市是：", ("纽约",), "The headquarters of the United Nations is in:", ("New York City", "New York")),
    ("fact", "长城所在的国家是：", ("中国",), "The Great Wall is located in:", ("China",)),
    ("fact", "地球的天然卫星是：", ("月球",), "Earth's natural satellite is the:", ("Moon",)),
    ("fact", "被称为红色星球的行星是：", ("火星",), "The planet known as the Red Planet is:", ("Mars",)),
    ("fact", "太阳系中最大的行星是：", ("木星",), "The largest planet in the Solar System is:", ("Jupiter",)),
    ("fact", "最小的质数是：", ("2", "二"), "The smallest prime number is:", ("2",)),
    ("fact", "奥林匹克五环的环数是：", ("5", "五"), "The number of Olympic rings is:", ("5", "five")),
    ("science", "水的化学式是：", ("H2O", "H₂O"), "The chemical formula for water is:", ("H2O", "H₂O")),
    ("science", "二氧化碳的化学式是：", ("CO2", "CO₂"), "The chemical formula for carbon dioxide is:", ("CO2", "CO₂")),
    ("science", "金的元素符号是：", ("Au",), "The chemical symbol for gold is:", ("Au",)),
    ("science", "氧的元素符号是：", ("O",), "The chemical symbol for oxygen is:", ("O",)),
    ("science", "氯化钠的常用化学式是：", ("NaCl",), "The common chemical formula for sodium chloride is:", ("NaCl",)),
    ("science", "速度的国际单位是：", ("米每秒", "m/s"), "The SI unit of speed is:", ("metre per second", "meter per second", "m/s")),
    ("science", "力的国际单位是：", ("牛顿", "N"), "The SI unit of force is the:", ("newton",)),
    ("science", "能量的国际单位是：", ("焦耳", "J"), "The SI unit of energy is the:", ("joule",)),
    ("science", "功率的国际单位是：", ("瓦特", "W"), "The SI unit of power is the:", ("watt",)),
    ("science", "电流的国际单位是：", ("安培", "A"), "The SI unit of electric current is the:", ("ampere",)),
    ("science", "热力学温度的国际单位是：", ("开尔文", "K"), "The SI unit of thermodynamic temperature is the:", ("kelvin",)),
    ("science", "人体心脏的心腔数量是：", ("4", "四"), "The number of chambers in the human heart is:", ("4", "four")),
    ("science", "标准大气压下水的冰点（摄氏度）是：", ("0", "0°C", "零"), "At standard pressure, water freezes at this Celsius temperature:", ("0", "0°C")),
    ("science", "标准大气压下水的沸点（摄氏度）是：", ("100", "100°C", "一百"), "At standard pressure, water boils at this Celsius temperature:", ("100", "100°C")),
    ("science", "植物光合作用通常吸收的气体是：", ("二氧化碳", "CO2", "CO₂"), "The gas plants normally absorb during photosynthesis is:", ("carbon dioxide", "CO2", "CO₂")),
    ("science", "植物光合作用通常释放的气体是：", ("氧气", "O2", "O₂"), "The gas plants normally release during photosynthesis is:", ("oxygen", "O2", "O₂")),
    ("science", "地球绕其运行的恒星是：", ("太阳",), "The star that Earth orbits is the:", ("Sun",)),
    ("science", "中性水溶液在常温下的 pH 值是：", ("7", "七"), "The approximate pH of a neutral aqueous solution at room temperature is:", ("7", "seven")),
    ("science", "氢元素的原子序数是：", ("1", "一"), "The atomic number of hydrogen is:", ("1", "one")),
    ("science", "碳元素的原子序数是：", ("6", "六"), "The atomic number of carbon is:", ("6", "six")),
    ("science", "平面三角形内角和的度数是：", ("180", "180°", "一百八十"), "The sum of the interior angles of a plane triangle in degrees is:", ("180", "180 degrees")),
    ("science", "细胞中常被称为能量工厂的细胞器是：", ("线粒体",), "The organelle often called the powerhouse of the cell is the:", ("mitochondrion", "mitochondria")),
    ("science", "二进制数 1010 对应的十进制数是：", ("10", "十"), "The decimal value of the binary number 1010 is:", ("10", "ten")),
    ("science", "真空中光速的精确值（米每秒）是：", ("299792458",), "The exact speed of light in vacuum in metres per second is:", ("299792458", "299,792,458")),
    ("science", "DNA 的中文全称是：", ("脱氧核糖核酸",), "DNA stands for:", ("deoxyribonucleic acid",)),
)


def _case(
    *,
    case_id: str,
    language: str,
    capability: str,
    raw_prompt: str,
    chat_prompt: str,
    expected: tuple[str, ...],
    raw_match_mode: str = "prefix",
    chat_match_mode: str = "prefix",
    language_match_mode: str = "auto",
) -> dict[str, Any]:
    if len(chat_prompt) < 48:
        chat_prompt = chat_prompt + (
            " 不要解释，不要重复题目，也不要添加单位、标点或其他无关文字。"
            if language == "zh"
            else " Do not explain, repeat the question, or add any unrelated text."
        )
    return {
        "id": case_id,
        "language": language,
        "capability": capability,
        "raw_prompt": raw_prompt,
        "chat_prompt": chat_prompt,
        "expected_any": list(expected),
        "raw_match_mode": raw_match_mode,
        "chat_match_mode": chat_match_mode,
        "language_match_mode": language_match_mode,
    }


def build_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language in ("zh", "en"):
        zh = language == "zh"
        for index in range(200):
            left = 137 + index * 17
            right = 29 + (index * 31) % 997
            expression = f"{left} + {right}"
            lead = "请独立计算下面的整数加法，只在等号后写最终数字，不要解释或重复题目：" if zh else "Compute this integer addition independently and write only the final number after the equals sign, with no explanation: "
            rows.append(_case(case_id=f"{language}_math_add_{index:03d}", language=language, capability="math", raw_prompt=f"{lead}{expression} =", chat_prompt=(f"请计算 {expression}，只回答最终数字。" if zh else f"Calculate {expression}. Answer with the final number only."), expected=(str(left + right),)))
        for index in range(150):
            right = 41 + (index * 19) % 701
            left = right + 113 + index * 13
            expression = f"{left} - {right}"
            lead = "请独立计算下面的整数减法，只在等号后写最终数字，不要解释或重复题目：" if zh else "Compute this integer subtraction independently and write only the final number after the equals sign, with no explanation: "
            rows.append(_case(case_id=f"{language}_math_sub_{index:03d}", language=language, capability="math", raw_prompt=f"{lead}{expression} =", chat_prompt=(f"请计算 {expression}，只回答最终数字。" if zh else f"Calculate {expression}. Answer with the final number only."), expected=(str(left - right),)))
        for index in range(150):
            left = 2 + (index % 37)
            right = 3 + ((index * 7) % 41)
            expression = f"{left} × {right}" if zh else f"{left} * {right}"
            lead = "请独立计算下面的整数乘法，只在等号后写最终数字，不要解释或重复题目：" if zh else "Compute this integer multiplication independently and write only the final number after the equals sign, with no explanation: "
            rows.append(_case(case_id=f"{language}_math_mul_{index:03d}", language=language, capability="math", raw_prompt=f"{lead}{expression} =", chat_prompt=(f"请计算 {expression}，只回答最终数字。" if zh else f"Calculate {expression}. Answer with the final number only."), expected=(str(left * right),)))
        for index in range(100):
            start = -37 + index * 3
            delta = 2 + index % 13
            values = [start + delta * offset for offset in range(4)]
            rendered = ", ".join(str(value) for value in values)
            raw = (f"观察下面这个等差整数数列的固定步长，并在冒号后只写下一项，不作解释：{rendered}，下一项是：" if zh else f"Identify the constant step in this arithmetic integer sequence and write only the next term after the colon: {rendered}; next term: ")
            chat = (f"数列 {rendered} 的下一项是什么？只回答数字。" if zh else f"What is the next term of {rendered}? Answer with the number only.")
            rows.append(_case(case_id=f"{language}_math_sequence_{index:03d}", language=language, capability="math", raw_prompt=raw, chat_prompt=chat, expected=(str(values[-1] + delta),)))
        for index in range(100):
            value = 1001 + index * 37
            expected = ("偶数",) if value % 2 == 0 else ("奇数",)
            if not zh:
                expected = ("even",) if value % 2 == 0 else ("odd",)
            raw = (f"判断下面这个整数的奇偶性，并在冒号后只写“奇数”或“偶数”，不要给出推导：整数 {value} 的类型是：" if zh else f"Classify the parity of this integer and write only 'odd' or 'even' after the colon, without reasoning: integer {value} is: ")
            chat = (f"整数 {value} 是奇数还是偶数？只回答分类。" if zh else f"Is the integer {value} odd or even? Answer with the classification only.")
            rows.append(_case(case_id=f"{language}_logic_parity_{index:03d}", language=language, capability="logic", raw_prompt=raw, chat_prompt=chat, expected=expected))
        for index in range(100):
            left = 101 + index * 43
            right = 307 + (index * 67) % 4001
            if left == right:
                right += 1
            answer = max(left, right)
            raw = (f"比较下面两个不同的整数，并在冒号后只写较大的那个数字，不要附加任何解释：{left} 与 {right} 中较大的是：" if zh else f"Compare these two distinct integers and write only the larger number after the colon, with no explanation: larger of {left} and {right}: ")
            chat = (f"{left} 和 {right} 哪个更大？只回答较大的数字。" if zh else f"Which is larger, {left} or {right}? Answer with the larger number only.")
            rows.append(_case(case_id=f"{language}_logic_compare_{index:03d}", language=language, capability="logic", raw_prompt=raw, chat_prompt=chat, expected=(str(answer),)))
        operations = ("add", "multiply", "divisible", "minimum")
        for index in range(100):
            operation = operations[index // 25]
            constant = 2 + index % 25
            name = f"case_{language}_{index:03d}"
            if operation == "add":
                body = (f"x + {constant}", f"x+{constant}")
                instruction = f"return its input plus {constant}"
                instruction_zh = f"返回输入值加 {constant} 的结果"
            elif operation == "multiply":
                body = (f"x * {constant}", f"x*{constant}")
                instruction = f"return its input multiplied by {constant}"
                instruction_zh = f"返回输入值乘以 {constant} 的结果"
            elif operation == "divisible":
                body = (f"x % {constant} == 0", f"x%{constant}==0")
                instruction = f"return whether its input is divisible by {constant}"
                instruction_zh = f"返回输入值能否被 {constant} 整除"
            else:
                body = (f"max(x, {constant})", f"max(x,{constant})")
                instruction = f"return the greater of its input and {constant}"
                instruction_zh = f"返回输入值与 {constant} 中较大的一个"
            raw = (f"补全下面这个 Python 函数，使它{instruction_zh}；只补写 return 后面的表达式：\ndef {name}(x):\n    return" if zh else f"Complete this Python function so it will {instruction}; write only the expression after return:\ndef {name}(x):\n    return")
            chat = (f"写出 Python 函数 {name}(x)，使它{instruction_zh}。只输出函数代码。" if zh else f"Write Python function {name}(x) that will {instruction}. Output code only.")
            chat_expected = tuple(f"return {item}" for item in body)
            rows.append({**_case(case_id=f"{language}_code_{index:03d}", language=language, capability="code", raw_prompt=raw, chat_prompt=chat, expected=body, raw_match_mode="prefix", chat_match_mode="contains", language_match_mode="exempt"), "chat_expected_any": list(chat_expected)})

    for fact_index, (capability, zh_prompt, zh_expected, en_prompt, en_expected) in enumerate(_FACTS):
        for variant in range(2):
            zh_lead = ("请补全下面这条基础知识陈述，只写出缺失的名称或数值，不要解释或重复题目：" if variant == 0 else "根据稳定的通用基础知识，在冒号后直接填写唯一答案，不添加任何其他文字：")
            en_lead = ("Complete this stable basic-knowledge statement with only the missing name or value and no explanation: " if variant == 0 else "Using stable general knowledge, write the single answer after the colon and nothing else: ")
            rows.append(_case(case_id=f"zh_{capability}_{fact_index:02d}_{variant}", language="zh", capability=capability, raw_prompt=zh_lead + zh_prompt, chat_prompt=f"{zh_prompt} 只回答答案。", expected=zh_expected))
            rows.append(_case(case_id=f"en_{capability}_{fact_index:02d}_{variant}", language="en", capability=capability, raw_prompt=en_lead + en_prompt, chat_prompt=f"{en_prompt} Answer only.", expected=en_expected))

    if len(rows) != EXPECTED_CASES:
        raise AssertionError(f"expected {EXPECTED_CASES} cases, built {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("generation suite contains duplicate ids")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    root = Path(__file__).resolve().parents[3]
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def build_suite(*, suite_path: Path, signatures_path: Path, report_path: Path) -> dict[str, Any]:
    rows = build_cases()
    signatures = [
        {"id": f"{row['id']}:{kind}", "benchmark": "generation_quality_v2", "prompt": str(row[f"{kind}_prompt"])}
        for row in rows
        for kind in ("raw", "chat")
    ]
    _write_jsonl(suite_path, rows)
    _write_jsonl(signatures_path, signatures)
    by_language = Counter(str(row["language"]) for row in rows)
    by_capability = Counter(str(row["capability"]) for row in rows)
    report = {
        "schema": REPORT_SCHEMA,
        "suite_schema": SUITE_SCHEMA,
        "status": "complete",
        "case_count": len(rows),
        "signature_count": len(signatures),
        "by_language": dict(sorted(by_language.items())),
        "by_capability": dict(sorted(by_capability.items())),
        "suite_path": _portable_path(suite_path),
        "suite_sha256": _sha256(suite_path),
        "decontamination_signatures_path": _portable_path(signatures_path),
        "decontamination_signatures_sha256": _sha256(signatures_path),
        "construction": "deterministic_project_authored_templates_and_stable_facts",
    }
    write_json_atomic(report_path, report, ensure_ascii=False, sort_keys=True, make_parents=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Build the pinned 2,000-case bilingual raw-generation suite.")
    parser.add_argument("--suite", default=str(root / "configs/eval/generation_quality.jsonl"))
    parser.add_argument("--signatures", default=str(root / "configs/eval/generation_quality_decontamination.jsonl"))
    parser.add_argument("--report", default=str(root / "configs/eval/generation_quality_report.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_suite(suite_path=Path(args.suite), signatures_path=Path(args.signatures), report_path=Path(args.report))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
