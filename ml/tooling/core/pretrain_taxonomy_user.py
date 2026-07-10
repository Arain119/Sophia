from __future__ import annotations

"""
User-facing fine-grained taxonomy for pretrain corpus analysis/sampling.

This taxonomy is **heuristic** (regex/rules) and intended for:
  - token-level supply accounting (by label)
  - building a balanced pretrain mix by down/up-sampling labels

It is NOT meant to be a perfect topic classifier.
"""

import re


LABELS: tuple[str, ...] = (
    # Code
    "code_algorithm",
    "code_application",
    "code_scripting",
    "code_config",
    # Math
    "math_derivation",
    "math_calculation",
    "math_logic_puzzle",
    # Encyclopedia / facts
    "encyclopedia_core",
    "encyclopedia_qa",
    # Academic / education
    "academic_textbook",
    "academic_paper",
    "professional_report",
    # News
    "news_mainstream",
    "news_commentary",
    # Translation / linguistics
    "translation_parallel",
    "linguistics_grammar",
    # Articles / blogs
    "article_editorial",
    "article_blog",
    # Literature
    "literature_classic",
    "literature_web",
    "literature_classical",
    # Daily comm
    "comm_conversational",
    # STEM / tech
    "stem_physics_math",
    "stem_engineering",
    "tech_blog_forum",
    # Finance
    "finance_macro",
    "finance_market",
    "finance_banking",
    # Humanities & social sciences
    "humanities_history",
    "humanities_philosophy",
    "social_psychology",
    # Government
    "gov_policy",
    "gov_exam",
    # Legal / medical
    "legal_statute",
    "legal_case",
    "medical_guideline",
    "medical_qa",
    # Restricted / other
    "nsfw",
    "boiler",
    "other",
)


# -------------------------
# Common / noise
# -------------------------


_RE_BOILER = re.compile(
    r"(免责声明|版权(所有|声明)|版权所有|ICP(备案)?|Copyright|All\s+Rights\s+Reserved)",
    re.IGNORECASE,
)

_RE_NSFW = re.compile(
    r"(无码|AV片|成人视频|色情|黄文|强奸|乱伦|兽交|幼女|未成年.*(性|性交)|"
    r"阴茎|阴道|乳房|口交|肛交|高潮|射精|性奴|SM调教|"
    r"\b(porn|porno|xxx|sex\s+video|hardcore)\b)",
    re.IGNORECASE,
)


# -------------------------
# Code
# -------------------------


_RE_CODE_ANY = re.compile(
    r"```|Traceback \(most recent call last\)|\bException\b|"
    # Avoid over-triggering on common English words (e.g. "from", "return", "class").
    # Prefer structural code cues instead.
    r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(|"
    r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\b|"
    r"\bfrom\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\s+[A-Za-z_*][A-Za-z0-9_*,\s]*\b|"
    r"\bimport\s+[A-Za-z_][A-Za-z0-9_.]*\b|"
    r"\b(async\s+def|await\s+|lambda\s+)\b|"
    r"\b(public|private|protected|static|void|int|float|double|String)\b|"
    r"#include\b|using\s+namespace\b|std::|console\.log\b|System\.out\.println\b|"
    r"\b(python|java(script)?|typescript|c\+\+|rust|golang|c#|sql|bash|powershell|node\.js|react|vue|docker|kubernetes|linux|git)\b",
    re.IGNORECASE,
)

_RE_CODE_ALGO = re.compile(
    r"\b(leetcode|lintcode|codeforces|atcoder|acm|oj)\b|力扣|题解|算法|复杂度|时间复杂度|空间复杂度|"
    r"\b(O\([^)]+\))\b|动态规划|贪心|二分|并查集|线段树|树状数组|最短路|拓扑排序|"
    r"\b(BFS|DFS|DP|Dijkstra|Floyd|KMP|Trie)\b",
    re.IGNORECASE,
)

_RE_CODE_SCRIPT = re.compile(
    r"^#!\s*/(usr/)?bin/(env\s+)?(bash|sh|zsh|python)\b|"
    r"\b(chmod|chown|systemctl|journalctl|apt(-get)?|yum|dnf|brew|pip|conda|kubectl|docker)\b|"
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE)\b|"
    r"\b(Get-ChildItem|Set-Location|Write-Output|New-Item|Invoke-WebRequest)\b",
    re.IGNORECASE | re.MULTILINE,
)

_RE_DOCKERFILE = re.compile(
    r"^\s*(FROM|RUN|CMD|COPY|ADD|ENV|WORKDIR|ENTRYPOINT|EXPOSE|ARG|LABEL)\b",
    re.IGNORECASE | re.MULTILINE,
)
_RE_JSON_KV = re.compile(r"\"[^\"]{1,64}\"\s*:\s*", re.IGNORECASE)
_RE_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:_-]*(\s+[^<>]{0,200})?>")
_RE_XML_DECL = re.compile(r"<\?xml\b", re.IGNORECASE)
_RE_YAML_KV_LINE = re.compile(
    r"^\s*[A-Za-z0-9_.-]{1,64}\s*:\s*[^:\n]{0,200}$", re.MULTILINE
)
_RE_INI_SECTION = re.compile(r"^\s*\[[^\]\n]{1,80}\]\s*$", re.MULTILINE)
_RE_INI_KV = re.compile(r"^\s*[A-Za-z0-9_.-]{1,64}\s*=\s*[^=\n]{0,200}$", re.MULTILINE)


def _looks_like_code_config(t: str) -> bool:
    s = (t or "").strip()
    if not s:
        return False

    if _RE_DOCKERFILE.search(s):
        return True
    if _RE_XML_DECL.search(s):
        return True

    # JSON-ish: starts with { or [, has several "k": occurrences.
    if s[:1] in "{[" and _RE_JSON_KV.search(s):
        if s.count("{") + s.count("}") + s.count("[") + s.count("]") >= 10:
            return True

    # YAML/INI: enough key-value lines suggests config.
    yaml_hits = len(_RE_YAML_KV_LINE.findall(s[:20000]))
    if yaml_hits >= 8:
        return True
    if (
        len(_RE_INI_SECTION.findall(s[:20000])) >= 2
        and len(_RE_INI_KV.findall(s[:20000])) >= 6
    ):
        return True

    # HTML markup-heavy (avoid false positives on a few tags).
    html_hits = len(_RE_HTML_TAG.findall(s[:20000]))
    if html_hits >= 18:
        return True
    return False


# -------------------------
# Math
# -------------------------


_RE_MATH_ANY = re.compile(
    r"(\$[^$\n]{6,}\$|\\\([^\\\n]{6,}\\\)|\\\[[^\\\n]{6,}\\\]|"
    r"\\(frac|sum|int|sqrt|pi|infty|begin|end)\b|"
    r"[≤≥≈≠∈∉⊂⊆⊇∪∩∑∫√π∞]|"
    r"(求解|证明|推导|方程|不等式|积分|导数|微分|极限|矩阵|向量|几何|代数|概率|数列|线性代数|微积分|数学题)|"
    r"\b(sin|cos|tan|sqrt|log|algebra|geometry|calculus|derivative|integral|matrix|vector|probability|equation)\b)",
    re.IGNORECASE,
)

_RE_MATH_DERIV = re.compile(
    r"(定理|引理|推论|命题|证明|proof|theorem|lemma|corollary)|"
    r"(\\begin\{(align|equation|gather|proof)\}|\\qed\b)",
    re.IGNORECASE,
)

_RE_MATH_PUZZLE = re.compile(
    r"(逻辑推理|行测|推理题|智力题|脑筋急转弯|真题|选择题|判断题|甲乙丙丁|"
    r"下列(说法|选项)|正确的是|不正确的是|答案解析)|"
    r"(^|\n)\s*[ABCD][\.\、]\s*",
    re.IGNORECASE,
)


# -------------------------
# Legal / medical
# -------------------------


_RE_LEGAL_ANY = re.compile(
    r"(法律|律师|合同|法院|判决|起诉|诉讼|刑法|民法|劳动法|条款|证据|法规|合规|侵权|知识产权|著作权|商标|专利|法典|司法)|"
    r"\b(contract|lawsuit|court|legal|law|attorney|copyright|patent)\b",
    re.IGNORECASE,
)

_RE_LEGAL_STATUTE = re.compile(
    r"(中华人民共和国|本法|条例|办法|规定|实施细则|司法解释|附则|"
    r"第[一二三四五六七八九十百千0-9]{1,6}条|"
    r"第[一二三四五六七八九十百千0-9]{1,6}章|"
    r"《[^》]{2,40}(法|条例|规定)》)",
    re.IGNORECASE,
)

_RE_LEGAL_CASE = re.compile(
    r"(判决书|裁定书|调解书|本院认为|本案|案号|审理|原告|被告|上诉人|被上诉人|"
    r"诉讼请求|事实与理由|辩称|代理人|二审|一审|再审)",
    re.IGNORECASE,
)

_RE_MED_ANY = re.compile(
    r"(症状|用药|剂量|诊断|处方|药物|发烧|咳嗽|血压|糖尿病|高血压|感染|肿瘤|"
    r"疼痛|过敏|急诊|体温|检查|临床|疗效|不良反应|治疗方案)|"
    r"\b(symptom|diagnos|dose|medicine|hospital|doctor|clinical|guideline)\b",
    re.IGNORECASE,
)

_RE_MED_GUIDELINE = re.compile(
    r"(指南|共识|诊疗(规范|指南|路径)?|推荐(意见|等级)?|循证|证据等级|"
    r"适应证|禁忌证|用法用量|不良反应|注意事项|临床试验|随机对照|"
    r"\b(ICD-?10|RCT|meta-?analysis)\b)",
    re.IGNORECASE,
)

_RE_MED_QA = re.compile(
    r"(医生(您好|好)|请问|我(该|应该)怎么办|可以(吃|用)什么药|有没有(关系|问题)|"
    r"多久能好|需要(检查|就医)吗|是不是|严重吗)",
    re.IGNORECASE,
)


# -------------------------
# Finance
# -------------------------


_RE_FIN_MACRO_STRONG = re.compile(
    r"\b(GDP|CPI|PPI|PMI|M2|LPR|FOMC|FED)\b|"
    r"(宏观|通胀|通货膨胀|货币政策|财政政策|降准|加息|降息|央行|美联储|"
    r"经济增长|失业率|汇率|外汇|国债收益率|利差|贸易顺差|赤字)",
    re.IGNORECASE,
)
_RE_FIN_RATE = re.compile(r"(利率|收益率)", re.IGNORECASE)
_RE_FIN_RATE_CTX = re.compile(
    r"(央行|美联储|货币政策|LPR|降准|加息|降息|国债)", re.IGNORECASE
)
_RE_FIN_MARKET = re.compile(
    r"(研报|证券|股票|股价|市值|估值|PE\b|PB\b|EPS\b|ROE\b|财报|业绩|净利润|营收|"
    r"现金流|投资建议|目标价|风险提示|机构观点|基金|债券|期货|行情)",
    re.IGNORECASE,
)
_RE_FIN_BANK = re.compile(
    r"(银行(业)?|存款|贷款|利息|理财|信用卡|征信|房贷|按揭|授信|风控|"
    r"不良贷款|资产负债表|净息差|资本充足率)",
    re.IGNORECASE,
)


# -------------------------
# Government / public admin
# -------------------------


_RE_GOV_EXAM = re.compile(
    r"(公务员(考试)?|国考|省考|事业单位|申论|行测|面试真题|备考|答案解析|范文|材料作文)",
    re.IGNORECASE,
)
_RE_GOV_POLICY = re.compile(
    r"(国务院|中央(委员会)?|人民政府|政府工作报告|白皮书|"
    r"发改委|工业和信息化部|工信部|教育部|财政部|民政部|商务部|"
    r"人民银行|央行|证监会|银保监会)|"
    r"(关于[^\n]{0,60}(通知|意见|决定|公告|实施方案|工作要点|规划|纲要))|"
    r"(〔\\d{4}〕\\d{1,4}号)",
    re.IGNORECASE,
)


# -------------------------
# STEM / tech (non-code)
# -------------------------


_RE_STEM_PHYSICS = re.compile(
    r"(量子|相对论|薛定谔|玻色|费米|标准模型|弦论|暗物质|暗能量|宇宙学|粒子物理|"
    r"凝聚态|热力学|电动力学|场论|拉格朗日|哈密顿量)|"
    r"\b(quantum|relativity|cosmology|particle\s+physics|thermodynamics|electrodynamics)\b",
    re.IGNORECASE,
)
_RE_STEM_ENGINEERING = re.compile(
    r"(GB/T|ISO|IEC|JIS|ASTM)|"
    r"(机械(工程)?|电气(工程)?|航天|航空|发动机|涡轮|材料力学|结构力学|电路|控制(系统)?|PLC)",
    re.IGNORECASE,
)
_RE_TECH_FORUM = re.compile(
    r"\b(stack\s*overflow|stackexchange|hacker\s*news|news\.ycombinator|v2ex|reddit)\b|"
    r"(论坛|社区|帖子|楼主|回复|评论区|置顶|更新日志|issue\s*#|pull\s+request)",
    re.IGNORECASE,
)


# -------------------------
# Humanities & social sciences
# -------------------------


_RE_HIST = re.compile(
    r"(朝代|皇帝|年号|史记|资治通鉴|编年史|传记|王朝|"
    r"公元前|公元|年代|考古|战争|起义|革命)",
    re.IGNORECASE,
)
_RE_PHILO = re.compile(
    r"(哲学|形而上学|认识论|伦理学|道德哲学|存在主义|现象学|逻辑学|"
    r"康德|黑格尔|尼采|柏拉图|亚里士多德|维特根斯坦)|"
    r"\b(philosophy|ethics|metaphysics|epistemology)\b",
    re.IGNORECASE,
)
_RE_SOC_PSY = re.compile(
    r"(心理学|社会学|人格|认知|行为|情绪|抑郁|焦虑|"
    r"实验|问卷|样本|量表|统计显著|相关性|回归分析)|"
    r"\b(psychology|sociology|cognitive|behavioral)\b",
    re.IGNORECASE,
)


# -------------------------
# Translation / linguistics
# -------------------------


_RE_TRANSLATION_ANY = re.compile(
    r"(翻译|译文|原文|中译英|英译中|译为|译成|translate|translation|interpret)",
    re.IGNORECASE,
)
_RE_LANG_GRAMMAR = re.compile(
    r"(词性|音标|释义|例句|语法|时态|从句|主谓|宾语|"
    r"\b(pronunciation|grammar|tense|clause|synonym|antonym)\b)",
    re.IGNORECASE,
)

_RE_PARALLEL_HINT = re.compile(
    r"(原文[:：].{0,30}\n.{0,10}(译文|翻译)[:：])|"
    r"(^|\n)\s*(中文|英文|Chinese|English)\s*[:：]",
    re.IGNORECASE,
)

_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_HAS_LATIN = re.compile(r"[A-Za-z]")


def _looks_like_parallel(t: str) -> bool:
    s = (t or "").strip()
    if not s:
        return False
    if not _RE_TRANSLATION_ANY.search(s):
        return False
    if _RE_PARALLEL_HINT.search(s):
        return True
    # Tab-separated bilingual lines.
    if "\t" in s and len(s) >= 200:
        return True
    # Mixed-language translation content without explicit markers.
    if len(s) >= 200 and _RE_HAS_CJK.search(s) and _RE_HAS_LATIN.search(s):
        return True
    return False


# -------------------------
# Encyclopedia / academic / news / articles
# -------------------------


_RE_WIKI = re.compile(
    r"(维基百科|Wikipedia|wiki)|"
    r"(^|\n)\s*(参考文献|外部链接|参见|目录)\s*[:：]?\s*$|"
    r"(Category:|分类:)",
    re.IGNORECASE | re.MULTILINE,
)

_RE_QA_MARK = re.compile(r"(^|\n)\s*(问|Q)[:：]|(^|\n)\s*(答|A)[:：]", re.IGNORECASE)

_RE_ACAD_PAPER = re.compile(
    r"(\bAbstract\b|\bIntroduction\b|\bMethod(s)?\b|\bResults\b|\bDiscussion\b|\bConclusion\b|\bReferences\b)|"
    r"(doi\s*:\s*|arXiv\s*:\s*|ISSN|et\s+al\.)|"
    r"(^|\n)\s*(摘要|关键词|引言|参考文献)\s*[:：]",
    re.IGNORECASE | re.MULTILINE,
)
_RE_ACAD_TEXTBOOK = re.compile(
    r"(第[一二三四五六七八九十百千0-9]{1,6}章|本章(小结|要点)?|学习目标|例题|习题|课后(练习|习题)|"
    r"思考题|知识点|教学)",
    re.IGNORECASE,
)
_RE_PRO_REPORT = re.compile(
    r"(研究报告|行业研究|深度报告|白皮书|风险提示|投资建议|数据来源|图表目录|"
    r"报告摘要|核心观点|免责声明)",
    re.IGNORECASE,
)

_RE_NEWS_MAIN = re.compile(
    r"(新华社|人民网|央视|中新网|记者|报道|讯|日电|"
    r"\b(reuters|ap\s+news|bloomberg)\b)|"
    r"\b\d{4}年\d{1,2}月\d{1,2}日\b",
    re.IGNORECASE,
)
_RE_NEWS_COMMENT = re.compile(
    r"(时评|社论|评论员|评论|观点|我们认为|笔者认为|专栏|解读)",
    re.IGNORECASE,
)

_RE_ART_EDITORIAL = re.compile(
    r"(编者按|本刊|专栏|特约(撰稿|评论)|长文|深度|观点)|"
    r"\b(editorial|opinion|column)\b",
    re.IGNORECASE,
)
_RE_ART_BLOG = re.compile(
    r"(博客|公众号|本文|作者|发表于|更新于|转载|阅读原文|"
    r"原创|个人观点)|"
    r"\b(blog|medium\.com|substack)\b",
    re.IGNORECASE,
)


# -------------------------
# Literature / comm
# -------------------------


_RE_CLASSICAL_ZH = re.compile(r"(之乎者也|焉|矣|吾|汝|其|乃|遂|然也|者也)")

# Keep zh/en chapter markers separate so we don't accidentally classify
# English books as "web novel" (the web-novel heuristics are zh-centric).
_RE_CHAPTER_ZH = re.compile(r"第[一二三四五六七八九十百千0-9]{1,6}[章节回卷]")
_RE_CHAPTER_EN = re.compile(
    r"\b(?:chapter|book|part)\b\s+(?:\d{1,4}|[ivxlcdm]{1,8})\b"
    r"|\b(?:act|scene)\b\s+(?:\d{1,4}|[ivxlcdm]{1,8})\b",
    re.IGNORECASE,
)
_RE_CHAPTER = re.compile(
    rf"(?:{_RE_CHAPTER_ZH.pattern}|{_RE_CHAPTER_EN.pattern})", re.IGNORECASE
)
_RE_WEBNOVEL_META = re.compile(
    r"(作者有话说|求收藏|求推荐|打赏|月票|VIP|订阅|晋江|起点|纵横|番茄小说|"
    r"更新|完结感言|上架感言)",
    re.IGNORECASE,
)
_RE_DIALOGUE = re.compile(
    r"(“|”|「|」|『|』|说道|问道|答道|笑道|低声|沉默|喊道|"
    r"\b(said|asked|replied|whispered|murmured|shouted|yelled|cried)\b)",
    re.IGNORECASE,
)


def _looks_like_literature_classical(t: str) -> bool:
    s = (t or "").strip()
    if len(s) < 300:
        return False
    # Wenyan tends to have repeated function words and fewer modern dialogue cues.
    hits = len(_RE_CLASSICAL_ZH.findall(s[:4000]))
    if hits >= 12 and not _RE_DIALOGUE.search(s[:2000]):
        return True
    return False


def _looks_like_web_novel(t: str) -> bool:
    s = (t or "").strip()
    if len(s) < 800:
        return False
    if _RE_WEBNOVEL_META.search(s):
        return True
    # This branch is intentionally zh-only to avoid pulling in regular English books.
    if _RE_CHAPTER_ZH.search(s) and len(_RE_DIALOGUE.findall(s[:6000])) >= 3:
        return True
    return False


def _looks_like_narrative_book(t: str) -> bool:
    s = (t or "").strip()
    if len(s) < 1200:
        return False
    head = s[:8000]
    # Dialogue cues:
    # - CJK quote marks / "said/asked/..." verbs via `_RE_DIALOGUE`
    # - ASCII double quotes are common in English fiction
    dlg = len(_RE_DIALOGUE.findall(head)) + max(0, int(head.count('"')) // 2)
    if _RE_CHAPTER.search(s) and dlg >= 2:
        return True
    if dlg >= 8:
        return True
    return False


_RE_COMM = re.compile(
    r"(楼主|回复|评论|点赞|哈哈|233|吃瓜|围观|顶|沙发|板凳|@[\w\u4e00-\u9fff]{1,16})|"
    r"(^|\n)\s*>\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _looks_like_comm(t: str) -> bool:
    s = (t or "").strip()
    if not s:
        return False
    if len(s) <= 1200 and _RE_COMM.search(s):
        return True
    # Many short lines often indicates forum/chat logs.
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if 6 <= len(lines) <= 80:
        short = sum(1 for ln in lines if len(ln.strip()) <= 30)
        if short / float(len(lines)) >= 0.55 and _RE_COMM.search(s):
            return True
    return False


# -------------------------
# Public API
# -------------------------


def infer_user_taxonomy_label(text: str) -> str:  # noqa: C901
    """
    Infer one LABELS entry for the given raw document text.

    Order is intentionally opinionated: we prefer to over-tag domain/structured
    content early (nsfw/code/math/legal/medical/finance/...) rather than leave
    it in "other".
    """

    t = str(text or "")
    if not t.strip():
        return "other"

    # Restricted / noise first.
    if _RE_NSFW.search(t):
        return "nsfw"
    if _RE_BOILER.search(t):
        return "boiler"

    # Code (incl. configs) before everything else.
    if _looks_like_code_config(t):
        return "code_config"
    if _RE_CODE_ANY.search(t):
        if _RE_CODE_SCRIPT.search(t):
            return "code_scripting"
        if _RE_CODE_ALGO.search(t):
            return "code_algorithm"
        return "code_application"

    # Translation / linguistics early.
    if _looks_like_parallel(t):
        return "translation_parallel"
    if _RE_LANG_GRAMMAR.search(t) and _RE_TRANSLATION_ANY.search(t):
        return "linguistics_grammar"

    # Medical / legal (split).
    if _RE_MED_QA.search(t):
        return "medical_qa"
    if _RE_MED_ANY.search(t):
        if _RE_MED_GUIDELINE.search(t):
            return "medical_guideline"
        if _RE_MED_QA.search(t) or _RE_QA_MARK.search(t):
            return "medical_qa"
        return "medical_guideline"

    if _RE_LEGAL_ANY.search(t):
        if _RE_LEGAL_STATUTE.search(t):
            return "legal_statute"
        if _RE_LEGAL_CASE.search(t) or _RE_QA_MARK.search(t):
            return "legal_case"
        return "legal_statute"

    # Finance.
    if _RE_FIN_MACRO_STRONG.search(t) or (
        _RE_FIN_RATE.search(t) and _RE_FIN_RATE_CTX.search(t)
    ):
        return "finance_macro"
    if _RE_FIN_MARKET.search(t):
        return "finance_market"
    if _RE_FIN_BANK.search(t):
        return "finance_banking"

    # Government.
    if _RE_GOV_EXAM.search(t):
        return "gov_exam"
    if _RE_GOV_POLICY.search(t):
        return "gov_policy"

    # STEM / tech (non-code).
    if _RE_STEM_ENGINEERING.search(t):
        return "stem_engineering"
    if _RE_STEM_PHYSICS.search(t):
        return "stem_physics_math"
    if _RE_TECH_FORUM.search(t):
        return "tech_blog_forum"

    # Humanities.
    if _RE_SOC_PSY.search(t):
        return "social_psychology"
    if _RE_PHILO.search(t):
        return "humanities_philosophy"
    if _RE_HIST.search(t):
        return "humanities_history"

    # Encyclopedia / QA.
    if _RE_WIKI.search(t):
        return "encyclopedia_core"
    if _RE_QA_MARK.search(t):
        # Longer single Q/A tends to be knowledge-style; short ones are usually chat/forums.
        if len(t) >= 1200:
            return "encyclopedia_qa"

    # Academic / education.
    if _RE_ACAD_PAPER.search(t):
        return "academic_paper"
    if _RE_ACAD_TEXTBOOK.search(t):
        return "academic_textbook"
    if _RE_PRO_REPORT.search(t):
        return "professional_report"

    # Math (after academic signals, to avoid swallowing papers/reports).
    if _RE_MATH_ANY.search(t):
        if _RE_MATH_DERIV.search(t):
            return "math_derivation"
        if _RE_MATH_PUZZLE.search(t):
            return "math_logic_puzzle"
        return "math_calculation"

    # News / commentary.
    if _RE_NEWS_MAIN.search(t):
        if _RE_NEWS_COMMENT.search(t):
            return "news_commentary"
        return "news_mainstream"

    # Articles / blogs.
    if _RE_ART_EDITORIAL.search(t):
        return "article_editorial"
    if _RE_ART_BLOG.search(t):
        return "article_blog"

    # Literature.
    if _looks_like_literature_classical(t):
        return "literature_classical"
    if _looks_like_web_novel(t):
        return "literature_web"
    if _looks_like_narrative_book(t):
        return "literature_classic"

    # Daily comm last.
    if _looks_like_comm(t):
        return "comm_conversational"

    return "other"
