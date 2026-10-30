"""只按文本规则召回；不调用 embedding，也不判断命理条件已经成立。"""
import re
from datetime import datetime, timedelta, timezone

STEMS = "甲乙丙丁戊己庚辛壬癸"
BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
GANZHI = {STEMS[i % 10] + BRANCHES[i % 12] for i in range(60)}
TRANSLATION = str.maketrans("時運歲幹論財氣陰陽殺", "时运岁干论财气阴阳杀")
PAIR = re.compile(rf"([{STEMS}][{BRANCHES}已己])日([{STEMS}][{BRANCHES}已己])时")


def normalized(text):
    return re.sub(r"\s+", "", text.translate(TRANSLATION))


def canonical_pair(pair):
    # 仅纠正干支标签的地支位置，不改动正文。
    return tuple(p[0] + ("巳" if p[1] in "已己" else p[1]) for p in pair)


def resolve_retrieval_periods(question, previous=None):
    """显式日期优先；追问沿用上轮年份，首次默认北京时间今年。"""
    now = datetime.now(timezone(timedelta(hours=8)))
    years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20|21)\d{2})(?!\d)", question)]
    year_range = re.search(r"((?:19|20|21)\d{2})\s*[年至到—~～-]+\s*((?:19|20|21)\d{2})", question)
    if year_range:
        first, last = map(int, year_range.groups())
        if 0 <= last - first <= 60:
            years = list(range(first, last + 1))
    if not years:
        relative = [(word, offset) for word, offset in [("前年", -2), ("去年", -1), ("今年", 0), ("明年", 1), ("后年", 2)] if word in question]
        years = [now.year + offset for _, offset in relative] or (previous or {}).get("years", [now.year])
    month_names = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                   "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
    month_pattern = r"(?<!\d)(1[0-2]|0?[1-9]|十二|十一|十|[一二三四五六七八九])\s*月"
    months = [int(m) if m.isdigit() else month_names[m] for m in re.findall(month_pattern, question)]
    month_range = re.search(r"(1[0-2]|[1-9]|十二|十一|十|[一二三四五六七八九])月?\s*[至到—~～-]+\s*(1[0-2]|[1-9]|十二|十一|十|[一二三四五六七八九])月", question)
    if month_range:
        a, b = [int(m) if m.isdigit() else month_names[m] for m in month_range.groups()]
        if a <= b:
            months = list(range(a, b + 1))
    if any(word in question for word in ["各月", "每月", "逐月", "各个月", "全年月份"]):
        months = list(range(1, 13))
    elif not months and any(word in question for word in ["本月", "这个月", "上月", "下月"]):
        offset = -1 if "上月" in question else 1 if "下月" in question else 0
        number = now.year * 12 + now.month - 1 + offset
        years, months = [number // 12], [number % 12 + 1]
    elif not months and "流月" in question:
        months = list(range(1, 13))
    return {"years": list(dict.fromkeys(years)), "months": list(dict.fromkeys(months))}


def collect_rule_chunks(collections, bazi, periods, year_fn, month_fn, question=""):
    pillars = str(bazi.get("四柱", "")).split()
    if len(pillars) != 4 or any(p not in GANZHI for p in pillars):
        raise ValueError("四柱须为按年、月、日、时排列的四个干支")
    natal_year, natal_month, day, hour = pillars
    years = periods["years"]
    months = periods["months"]
    annual = [year_fn(y).removesuffix("年") for y in years]
    monthly = [month_fn(y, m) for y in years for m in months]
    explicit_annual = re.findall(rf"([{STEMS}][{BRANCHES}])年", question)
    if explicit_annual:
        annual = list(dict.fromkeys(explicit_annual))
    monthly += [f"题目指定{gz}月" for gz in re.findall(rf"([{STEMS}][{BRANCHES}])月", question)]
    all_luck = any(word in question for word in ["全部大运", "所有大运", "各步大运", "每步大运", "一生大运"])
    luck = [x["干支"] for x in bazi.get("大运", []) if all_luck or any(x["开始年份"] <= y <= x["结束年份"] for y in years)]
    # 角色词是规则的一部分，单独出现一个干支不算年柱/流年命中。
    rules = [
        (f"年柱{natal_year}", rf"{natal_year}(?:年|岁)|(?:年柱|生年|年命)[^。；]{{0,6}}{natal_year}"),
        (f"月柱{natal_month}", rf"{natal_month}月|月柱[^。；]{{0,6}}{natal_month}"),
        (f"月令{natal_month[1]}", rf"[{BRANCHES}]*{natal_month[1]}[{BRANCHES}]*月|(?:月令|生于|生於){natal_month[1]}"),
        (f"日柱{day}", rf"{day}日|日柱[^。；]{{0,6}}{day}"),
        (f"日干{day[0]}", rf"(?:六)?{day[0]}日|(?:^|[。；]){day[0]}[木火土金水]|日(?:干|主)(?:为|為)?{day[0]}"),
    ]
    for gz in dict.fromkeys(luck):
        rules.append((f"大运{gz}", rf"(?:行|逢|交|入|大运|运至|运到|运走|运)[^。；]{{0,8}}{gz}(?![年月日时])|{gz}(?![年月日时])[^。；]{{0,4}}运"))
    # 十年大运的干支在原文中可能按天干、地支或五行方位分别讨论。
    elements = dict(zip(STEMS, "木木火火土土金金水水"))
    directions = {"木": "东", "火": "南", "金": "西", "水": "北", "土": "中央"}
    for gz in dict.fromkeys(luck):
        element = elements[gz[0]]
        rules.append((f"大运{gz}的干支/五行通论（仍需核对条件）",
                      rf"(?:行|逢|交|入|运至|运到)[^。；]{{0,4}}[{gz}{element}][^。；]{{0,3}}运|[{gz}]运|{element}运|{directions[element]}方?运"))
    for gz in dict.fromkeys(annual):
        rules.append((f"流年{gz}", rf"{gz}(?:年|岁)|(?:流年|太岁|岁君|岁运)[^。；]{{0,8}}{gz}"))
    for text in dict.fromkeys(monthly):
        gz = re.search(rf"([{STEMS}][{BRANCHES}])月", text).group(1)
        rules.append((f"流月{gz}", rf"{gz}月|(?:流月|月建)[^。；]{{0,6}}{gz}|[{BRANCHES}]*{gz[1]}[{BRANCHES}]*月"))
    # 只补充直接对应角色的通论；不会把其他日时章节当作通论。
    rules += [("年月柱通论", r"论年月日时|年为根|月为苗|月令乃提纲"),
              ("大运通论", r"论大运|大运者|运途论"),
              ("流年通论", r"论太岁|流年者|太岁者|岁运论")]
    if monthly:
        rules.append(("流月通论", r"论流月|流月者|月建者"))
    candidates = {}
    def add(book, source_id, text, reasons):
        key = (book, normalized(text))
        if key not in candidates:
            candidates[key] = {"book": book, "id": source_id, "text": text, "rules": []}
        candidates[key]["rules"] = list(dict.fromkeys(candidates[key]["rules"] + reasons))
    for book, collection in collections.items():
        records = collection.get(include=["documents"])
        for source_id, document in zip(records["ids"], records["documents"]):
            if not document:
                continue
            paragraphs = [p.strip() for p in re.split(rf"(?m)\n\s*\n|(?=^[{STEMS}][木火土金水]\s*$)", document.strip()) if p.strip()]
            if re.search(r"[.．…]{5,}", document):
                continue  # 排除目录点线；不改动库内原始数据。
            chapter_rules = [label for label, pattern in rules if label.endswith("通论") and re.search(pattern, normalized(paragraphs[0]))] if paragraphs[0].startswith("#") else []
            chapter = re.search(rf"六([{STEMS}])日([{STEMS}][{BRANCHES}已己])时断", normalized(paragraphs[0]))
            if chapter:
                chapter_hour = canonical_pair((day, chapter[2]))[1]
                if chapter[1] != day[0] or chapter_hour != hour:
                    continue
                general = next((p for p in paragraphs if normalized(p).startswith(f"{day[0]}日{hour}时")), "")
                for paragraph in paragraphs:
                    match = PAIR.match(normalized(paragraph))
                    if match and canonical_pair(match.groups()) == (day, hour):
                        text = paragraphs[0] + ("\n\n本章通论：\n" + general if general else "") + "\n\n日时专条：\n" + paragraph
                        add(book, source_id, text, [f"日柱{day}+时柱{hour}专条（巳/已/己仅在标签中兼容）"])
                continue
            for paragraph in paragraphs:
                if "通天論" in paragraph and "作者：" in paragraph:
                    paragraph = paragraph[paragraph.index("通天論"):]
                text = normalized(paragraph)
                if text.startswith(("来源：", "来源:", "http", "下载日期", "#")):
                    continue
                pairs = PAIR.findall(text)
                if pairs:
                    # 混有其他命例/日时条目的段落，不直接套给用户。
                    if all(canonical_pair(pair) == (day, hour) for pair in pairs):
                        add(book, source_id, paragraph, [f"日柱{day}+时柱{hour}专条"])
                    continue
                reasons = chapter_rules + [label for label, pattern in rules if re.search(pattern, text)]
                if re.search(r"六十甲子|纳音|納音", normalized(paragraphs[0])) and re.match(rf"{natal_year}[金木水火土]?[，,]", text):
                    reasons.append(f"年柱{natal_year}的六十甲子/纳音条目（非日主五行）")
                if reasons:
                    add(book, source_id, paragraph, reasons)
    return list(candidates.values()), {"目标年份": years, "目标月份": months, "流年": annual, "流月": monthly, "大运": luck}
