"""Описание признаков: русские подписи, типы, расшифровка кодов, группы.

Это единственный источник правды для интерфейса — фронтенд строит по нему
и анкету, и карточку студента. Сырые коды (`studytime=2`, `Mjob=at_home`)
конечному пользователю не показываются никогда.

Расшифровки взяты из условия задачи (таблицы колонок в ТЗ), русские подписи —
из словаря RU в ноутбуке.
"""
from __future__ import annotations

#: Предмет, на котором обучена модель. Без него вопрос «сколько раз вы уже
#: не сдавали этот предмет» повисает в воздухе: непонятно, про что отвечать.
SUBJECT = "португальский язык"

# Смысловые блоки для группировки в UI
GROUPS = [
    {"id": "study", "label": "Успеваемость и учебное поведение",
     "hint": "Как идёт учёба и сколько времени на неё уходит"},
    {"id": "personal", "label": "Личное",
     "hint": "Возраст, здоровье, свободное время"},
    {"id": "family", "label": "Семья",
     "hint": "Родители, опекун, состав семьи"},
    {"id": "grades", "label": "Оценки за контрольные",
     "hint": "Только для прогноза конца семестра (уровень L1)"},
]


def _scale(name, label, group, low, high, hint=None, labels=None):
    """Порядковая шкала 1-5 (или другая) — слайдер с текстовыми подписями концов."""
    return {"name": name, "label": label, "group": group, "kind": "scale",
            "min": low, "max": high, "hint": hint,
            "values": labels or {}, "default": (low + high) // 2}


def _int(name, label, group, low, high, default, hint=None):
    return {"name": name, "label": label, "group": group, "kind": "int",
            "min": low, "max": high, "default": default, "hint": hint, "values": {}}


def _choice(name, label, group, values, default, hint=None):
    """values: {код: человеческая подпись}"""
    return {"name": name, "label": label, "group": group, "kind": "choice",
            "values": values, "default": default, "hint": hint}


def _yesno(name, label, group, default="no", hint=None):
    return _choice(name, label, group, {"yes": "Да", "no": "Нет"}, default, hint)


EDU = {0: "нет образования", 1: "начальное (4 класса)", 2: "5–9 классов",
       3: "среднее специальное", 4: "высшее"}
JOB = {"teacher": "преподаватель", "health": "медицина", "services": "сфера услуг",
       "at_home": "не работает / дома", "other": "другое"}
STUDYTIME = {1: "менее 2 часов", 2: "2–5 часов", 3: "5–10 часов", 4: "более 10 часов"}
TRAVEL = {1: "менее 15 минут", 2: "15–30 минут", 3: "30–60 минут", 4: "больше часа"}
FAILURES = {0: "ни разу", 1: "один раз", 2: "два раза", 3: "три и более"}
LOW_HIGH = {1: "очень мало", 2: "мало", 3: "средне", 4: "много", 5: "очень много"}
BAD_GOOD = {1: "очень плохо", 2: "плохо", 3: "средне", 4: "хорошо", 5: "очень хорошо"}

FEATURES = [
    # ---------------- успеваемость и поведение ----------------
    _choice("failures", "Прошлые незачёты по предмету", "study", FAILURES, 0),
    _int("absences", "Пропущено занятий", "study", 0, 93, 4,
         "Количество пропущенных занятий за год"),
    _choice("studytime", "Самоподготовка в неделю", "study", STUDYTIME, 2),
    _choice("traveltime", "Дорога до учёбы", "study", TRAVEL, 1),
    _yesno("schoolsup", "Получает дополнительную поддержку от школы", "study"),
    _yesno("famsup", "Семья помогает с учёбой", "study", "yes"),
    _yesno("paid", "Ходит на платные занятия по предмету", "study"),
    _yesno("activities", "Занимается чем-то помимо учёбы", "study", "yes"),
    _yesno("higher", "Планирует учиться дальше", "study", "yes"),
    _yesno("nursery", "Ходил в детский сад", "study", "yes"),
    _choice("reason", "Почему выбрал это учебное заведение", "study",
            {"home": "близко к дому", "reputation": "репутация",
             "course": "из-за программы", "other": "другое"}, "course"),
    _choice("school", "Учебное заведение", "study",
            {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"}, "GP"),
    # ---------------- личное ----------------
    _int("age", "Возраст", "personal", 15, 22, 17),
    _choice("sex", "Пол", "personal", {"F": "женский", "M": "мужской"}, "F"),
    _choice("address", "Где живёт", "personal", {"U": "город", "R": "село"}, "U"),
    _scale("famrel", "Отношения в семье", "personal", 1, 5, labels=BAD_GOOD),
    _scale("freetime", "Свободное время после учёбы", "personal", 1, 5, labels=LOW_HIGH),
    _scale("goout", "Как часто встречается с друзьями", "personal", 1, 5, labels=LOW_HIGH),
    _scale("Dalc", "Алкоголь в будни", "personal", 1, 5, labels=LOW_HIGH),
    _scale("Walc", "Алкоголь в выходные", "personal", 1, 5, labels=LOW_HIGH),
    _scale("health", "Самооценка здоровья", "personal", 1, 5, labels=BAD_GOOD),
    _yesno("romantic", "Есть романтические отношения", "personal"),
    _yesno("internet", "Интернет дома", "personal", "yes"),
    # ---------------- семья ----------------
    _choice("Medu", "Образование матери", "family", EDU, 2),
    _choice("Fedu", "Образование отца", "family", EDU, 2),
    _choice("Mjob", "Работа матери", "family", JOB, "other"),
    _choice("Fjob", "Работа отца", "family", JOB, "other"),
    _choice("guardian", "Кто опекун", "family",
            {"mother": "мать", "father": "отец", "other": "другой"}, "mother"),
    _choice("famsize", "Размер семьи", "family",
            {"LE3": "до 3 человек", "GT3": "больше 3 человек"}, "GT3"),
    _choice("Pstatus", "Родители живут вместе", "family",
            {"T": "вместе", "A": "раздельно"}, "T"),
    # ---------------- оценки (только L1) ----------------
    _int("G1", "Балл за первую контрольную", "grades", 0, 20, 11,
         "Только для прогноза конца семестра"),
    _int("G2", "Балл за вторую контрольную", "grades", 0, 20, 11,
         "Только для прогноза конца семестра"),
]

#: Формулировки для анкеты студента (от второго лица). Там, где ключа нет,
#: берётся обычный `label`: карточка куратора и анкета — разные адресаты,
#: и «Ходит на платные занятия» о самом себе читается странно.
SELF_WORDING = {
    # --- учёба ---
    "failures":   (f"Сколько раз вы уже не сдавали {SUBJECT}?",
                   "Считаются только прошлые попытки по этому предмету"),
    "absences":   ("Сколько занятий вы пропустили за учебный год?",
                   "Примерно. Не помните точно — поставьте на глаз"),
    "studytime":  ("Сколько часов в неделю вы занимаетесь самостоятельно?",
                   "Домашние задания и подготовка вне занятий"),
    "higher":     ("Собираетесь ли вы продолжать учёбу после школы?",
                   "Колледж, университет — любое продолжение"),
    "paid":       (f"Ходите ли вы на платные занятия по предмету «{SUBJECT}»?",
                   "Репетитор или платные курсы"),
    "schoolsup":  ("Занимаетесь ли вы с преподавателем дополнительно?",
                   "Бесплатные дополнительные занятия от школы: консультации, "
                   "подготовка к контрольным"),
    "famsup":     ("Помогают ли вам дома с учёбой?",
                   "Родители или другие близкие"),
    # --- как проходит день ---
    "freetime":   ("Сколько у вас свободного времени после учёбы?", None),
    "goout":      ("Как часто вы проводите время с друзьями?", None),
    "activities": ("Занимаетесь ли вы чем-то помимо учёбы?",
                   "Спорт, кружки, музыка, работа"),
    "traveltime": ("Сколько времени вы добираетесь до учёбы?",
                   "В одну сторону"),
    "health":     ("Как вы оцениваете своё здоровье?", None),
    # --- о вас ---
    "age":        ("Сколько вам лет?", None),
    "sex":        ("Какой у вас пол?", None),
    "address":    ("Где вы живёте?", None),
    "famrel":     ("Как вы оцениваете отношения в семье?", None),
    # --- оценки (необязательный шаг) ---
    "G1":         ("Какой балл вы получили за первую контрольную?",
                   "По 20-балльной шкале. Зачёт — от 10 баллов"),
    "G2":         ("Какой балл вы получили за вторую контрольную?",
                   "По 20-балльной шкале. Зачёт — от 10 баллов"),
    # --- в анкете не спрашиваются, но подписи нужны карточке куратора ---
    "school":     ("Где вы учитесь?", None),
    "reason":     ("Почему вы выбрали это учебное заведение?", None),
    "nursery":    ("Ходили ли вы в детский сад?", None),
    "internet":   ("Есть ли у вас дома интернет?", None),
    "romantic":   ("Есть ли у вас романтические отношения?", None),
    "Dalc":       ("Как часто вы употребляете алкоголь в будни?", None),
    "Walc":       ("Как часто вы употребляете алкоголь в выходные?", None),
    "Medu":       ("Какое образование у вашей матери?", None),
    "Fedu":       ("Какое образование у вашего отца?", None),
    "Mjob":       ("Кем работает ваша мать?", None),
    "Fjob":       ("Кем работает ваш отец?", None),
    "guardian":   ("Кто ваш опекун?", None),
    "famsize":    ("Сколько человек в вашей семье?", None),
    "Pstatus":    ("Живут ли ваши родители вместе?", None),
}

BY_NAME = {f["name"]: f for f in FEATURES}

#: Признаки анкеты студента — уровень L2, без единой оценки по предмету.
SURVEY_FEATURES = [f["name"] for f in FEATURES if f["group"] != "grades"]

#: Анкета студента.
#:
#: Состав подобран не только по важности признаков. Замер показал, что заметно
#: влияют лишь три (`higher`, `failures`, `school`), но анкета из трёх вопросов
#: выглядит несерьёзно и не даёт человеку понять, что вообще учитывается.
#: Поэтому оставлены вопросы, на которые студент может ответить уверенно
#: и которые он ожидает увидеть в разговоре про учёбу.
#:
#: Убрано осознанно: алкоголь, романтические отношения, работа и образование
#: родителей, опекун, размер семьи, живут ли родители вместе, детский сад,
#: причина выбора школы. Первые — вторжение в личное без пользы для ответа,
#: остальные — предсказательный шум (у половины важность отрицательная).
#: `school` не спрашиваем: студенту нечего ответить про две португальские
#: школы, а рычагом он не является. Неспрошенное подставляется типичным
#: значением обучающей выборки.
SURVEY_CORE = [
    "failures", "absences", "studytime", "higher", "paid", "schoolsup", "famsup",
    "freetime", "goout", "activities", "traveltime", "health",
    "age", "sex", "address", "famrel",
]

#: Необязательный шаг: если контрольные уже написаны, модель уровня L1
#: заметно точнее (ROC-AUC 0.971 против 0.848).
SURVEY_GRADES = ["G1", "G2"]

SURVEY_STEPS_SHORT = [
    {"id": "study", "title": "Учёба", "features": [
        "failures", "absences", "studytime", "higher",
        "paid", "schoolsup", "famsup"]},
    {"id": "day", "title": "Как проходит день", "features": [
        "freetime", "goout", "activities", "traveltime", "health"]},
    {"id": "about", "title": "О вас", "features": [
        "age", "sex", "address", "famrel"]},
    {"id": "grades", "title": "Оценки", "optional": True, "features": [
        "G1", "G2"]},
]

#: Полный список (используется картой признаков куратора, не анкетой)
SURVEY_STEPS = [
    {"id": "study", "title": "Учёба", "features": [
        "failures", "absences", "studytime", "higher", "paid", "schoolsup",
        "famsup", "activities", "traveltime", "reason", "school", "nursery"]},
    {"id": "personal", "title": "О себе", "features": [
        "age", "sex", "address", "health", "freetime", "goout", "romantic",
        "internet", "Dalc", "Walc"]},
    {"id": "family", "title": "Семья", "features": [
        "Medu", "Fedu", "Mjob", "Fjob", "guardian", "famsize", "Pstatus", "famrel"]},
]


def decode(name: str, value):
    """Человеческая подпись значения. `studytime=2` -> «2–5 часов»."""
    f = BY_NAME.get(name)
    if f is None:
        return str(value)
    vals = f.get("values") or {}
    if not vals:
        return value
    for key in (value, str(value)):
        if key in vals:
            return vals[key]
    try:
        iv = int(float(value))
        if iv in vals:
            return vals[iv]
    except (TypeError, ValueError):
        pass
    return str(value)


def as_json():
    """Схема для /api/schema — фронтенд строит по ней анкету и карточку."""
    return {
        "groups": GROUPS,
        "features": [
            {**f,
             "values": {str(k): v for k, v in (f.get("values") or {}).items()},
             "self_label": SELF_WORDING.get(f["name"], (None, None))[0],
             "self_hint": SELF_WORDING.get(f["name"], (None, None))[1]}
            for f in FEATURES
        ],
        "survey_features": SURVEY_FEATURES,
        "survey_steps": SURVEY_STEPS,
        "subject": SUBJECT,
        "survey_core": SURVEY_CORE,
        "survey_grades": SURVEY_GRADES,
        "survey_steps_short": SURVEY_STEPS_SHORT,
        "required_l2": [f["name"] for f in FEATURES if f["group"] != "grades"],
        "grade_features": ["G1", "G2"],
    }
