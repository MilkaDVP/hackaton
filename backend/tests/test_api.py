"""Тесты API: happy path и все способы сломать загрузку файла."""
from __future__ import annotations

import io

FILE = "multipart/form-data"


def upload(client, content: bytes, name="students.csv", **data):
    return client.post("/api/predict/batch",
                       files={"file": (name, content, "text/csv")}, data=data)


# --------------------------------------------------------------------------
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_schema_drives_ui(client):
    s = client.get("/api/schema").json()
    names = {f["name"] for f in s["features"]}
    # анкета студента не содержит ни одной оценки по предмету
    assert not ({"G1", "G2", "G3"} & set(s["survey_features"]))
    # каждый шаг мастера ссылается на существующие признаки
    for step in s["survey_steps"]:
        assert set(step["features"]) <= names
    # у категориальных признаков есть человеческие подписи значений
    studytime = next(f for f in s["features"] if f["name"] == "studytime")
    assert studytime["values"]["2"] == "2–5 часов"


def test_model_info_has_both_levels(client):
    m = client.get("/api/model-info").json()
    assert set(m["metrics"]) == {"L1", "L2"}
    assert 0.0 < m["threshold"]["default"] < 1.0
    assert m["threshold"]["default"] != 0.5, "порог не должен быть дефолтным 0.5"


# --------------------------------------------------------------------------
def test_semicolon_file(client, por_csv):
    """Родной формат датасета: разделитель ';'."""
    r = upload(client, por_csv, "student-por.csv")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["summary"]["n_rows"] == 649
    assert j["file"]["delimiter"] == ";"
    assert len(j["rows"]) == 649


def test_comma_file(client, por_df):
    """Тот же файл, пересохранённый с запятой, обязан работать так же."""
    content = por_df.to_csv(index=False, sep=",").encode("utf-8")
    r = upload(client, content, "comma.csv")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["summary"]["n_rows"] == 649
    assert j["file"]["delimiter"] == ","


def test_tab_file(client, por_df):
    content = por_df.to_csv(index=False, sep="\t").encode("utf-8")
    r = upload(client, content, "tabs.tsv")
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["n_rows"] == 649


def test_cp1251_encoding(client, por_df):
    content = por_df.to_csv(index=False, sep=";").encode("cp1251")
    r = upload(client, content, "win.csv")
    assert r.status_code == 200, r.text


def test_xlsx(client, por_df):
    buf = io.BytesIO()
    por_df.head(50).to_excel(buf, index=False)
    r = client.post("/api/predict/batch",
                    files={"file": ("s.xlsx", buf.getvalue(),
                                    "application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet")})
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["n_rows"] == 50


# --------------------------------------------------------------------------
def test_level_auto_l1_when_grades_present(client, por_csv):
    j = upload(client, por_csv).json()
    assert j["level"] == "L1"
    assert "G1" in j["level_note"] or "оценк" in j["level_note"]


def test_level_auto_l2_without_grades(client, por_df):
    content = por_df.drop(columns=["G1", "G2"]).to_csv(index=False, sep=";").encode()
    j = upload(client, content, "no_grades.csv").json()
    assert j["level"] == "L2"


def test_level_override_l1_without_grades_is_422(client, por_df):
    content = por_df.drop(columns=["G1", "G2"]).to_csv(index=False, sep=";").encode()
    r = upload(client, content, "no_grades.csv", level="L1")
    assert r.status_code == 422
    assert "G1" in r.json()["error"]["message"]


# --------------------------------------------------------------------------
def test_g3_present_enables_verification(client, por_csv):
    j = upload(client, por_csv).json()
    assert "verification" in j
    assert j["verification"]["n"] == 649
    assert any("actual" in row for row in j["rows"])


def test_g3_never_reaches_features(client, por_df):
    """Ключевой тест утечки: G3 не должна влиять на вероятность."""
    without = por_df.drop(columns=["G3"])
    p_without = upload(client, without.to_csv(index=False, sep=";").encode()).json()

    # тот же файл, но G3 перевёрнута — вероятности обязаны совпасть побайтово
    flipped = por_df.copy()
    flipped["G3"] = 20 - flipped["G3"]
    p_flipped = upload(client, flipped.to_csv(index=False, sep=";").encode()).json()

    a = [r["probability"] for r in p_without["rows"]]
    b = [r["probability"] for r in p_flipped["rows"]]
    assert a == b, "G3 повлияла на предсказание — это утечка"

    # и её нет среди возвращаемых признаков
    assert "G3" not in p_flipped["rows"][0]["features"]


# --------------------------------------------------------------------------
def test_missing_columns_422_with_suggestion(client, por_df):
    broken = por_df.rename(columns={"failures": "failure", "absences": "absence"})
    r = upload(client, broken.to_csv(index=False, sep=";").encode(), "typo.csv")
    assert r.status_code == 422
    err = r.json()["error"]
    assert "failures" in err["details"]["missing"]
    # подсказка про опечатку
    assert err["details"]["suggestions"].get("failures") == "failure"


def test_extra_columns_ignored(client, por_df):
    extra = por_df.copy()
    extra["мусор"] = "x"
    extra["another"] = 1
    r = upload(client, extra.to_csv(index=False, sep=";").encode(), "extra.csv")
    assert r.status_code == 200
    assert "мусор" not in r.json()["rows"][0]["features"]


def test_empty_file(client):
    r = upload(client, b"", "empty.csv")
    assert r.status_code == 400
    assert "пуст" in r.json()["error"]["message"].lower()


def test_garbage_file(client):
    r = upload(client, b"\x00\x01\x02 not a table at all", "garbage.csv")
    assert r.status_code in (400, 422)
    assert "error" in r.json()


def test_header_only(client, por_df):
    content = (";".join(por_df.columns) + "\n").encode()
    r = upload(client, content, "header.csv")
    assert r.status_code == 400


def test_single_column_hints_delimiter(client):
    r = upload(client, b"onlyonecolumn\n1\n2\n", "one.csv")
    assert r.status_code == 400
    assert "разделител" in r.json()["error"]["hint"].lower()


# --------------------------------------------------------------------------
def test_rows_sorted_and_ranked(client, por_csv):
    rows = upload(client, por_csv).json()["rows"]
    by_rank = sorted(rows, key=lambda r: r["rank"])
    probs = [r["probability"] for r in by_rank]
    assert probs == sorted(probs, reverse=True), "ранг должен идти по убыванию риска"


def test_top_factors_are_human_readable(client, por_csv):
    rows = upload(client, por_csv).json()["rows"]
    factors = [f for r in rows for f in r["top_factors"]]
    assert factors, "объяснения должны быть"
    for f in factors[:50]:
        assert f["label"] and not f["label"].islower() or " " in f["label"]
        assert f["direction"] in ("up", "down")
        # сырых кодов быть не должно
        assert f["value_label"] not in ("at_home", "GT3", "LE3")


def test_school_flagged_as_not_a_lever(client, por_csv):
    rows = upload(client, por_csv).json()["rows"]
    school = [f for r in rows for f in r["top_factors"] if f["feature"] == "school"]
    if school:
        assert all(f["not_a_lever"] for f in school)


def test_custom_threshold_changes_shortlist(client, por_csv):
    low = upload(client, por_csv, threshold="0.1").json()
    high = upload(client, por_csv, threshold="0.9").json()
    assert low["summary"]["in_shortlist"] > high["summary"]["in_shortlist"]


def test_export_csv(client, por_csv):
    rows = upload(client, por_csv).json()["rows"][:20]
    r = client.post("/api/export", json={"rows": rows, "format": "csv"})
    assert r.status_code == 200
    text = r.content.decode("utf-8-sig")
    assert "probability" in text and "rank" in text


def test_export_xlsx(client, por_csv):
    rows = upload(client, por_csv).json()["rows"][:20]
    r = client.post("/api/export", json={"rows": rows, "format": "xlsx"})
    assert r.status_code == 200
    assert r.content[:2] == b"PK"


# --------------------------------------------------------------------------
def test_survey_happy_path(client):
    s = client.get("/api/schema").json()
    answers = {}
    for f in s["features"]:
        if f["name"] in s["survey_features"]:
            answers[f["name"]] = f["default"]
    r = client.post("/api/predict/single", json={"answers": answers})
    assert r.status_code == 200, r.text
    j = r.json()
    assert 0.0 <= j["probability"] <= 1.0
    assert j["risk_band"] in ("low", "medium", "high")
    assert j["disclaimer"]["not_a_verdict"]
    assert j["disclaimer"]["privacy"]


def test_survey_uses_grades_when_given(client):
    """Оценки необязательны, но если указаны обе — считаем моделью конца семестра."""
    s = client.get("/api/schema").json()
    core = {f["name"]: f["default"] for f in s["features"] if f["name"] in s["survey_core"]}

    without = client.post("/api/predict/single", json={"answers": core}).json()
    assert without["level"] == "L2"

    bad = client.post("/api/predict/single",
                      json={"answers": {**core, "G1": 5, "G2": 4}}).json()
    good = client.post("/api/predict/single",
                       json={"answers": {**core, "G1": 17, "G2": 18}}).json()
    assert bad["level"] == "L1" and good["level"] == "L1"
    assert bad["probability"] > good["probability"], "плохие оценки обязаны поднять риск"


def test_survey_one_grade_falls_back_to_l2(client):
    """Одной оценки мало для модели конца семестра — откатываемся на L2."""
    s = client.get("/api/schema").json()
    core = {f["name"]: f["default"] for f in s["features"] if f["name"] in s["survey_core"]}
    r = client.post("/api/predict/single", json={"answers": {**core, "G1": 12}}).json()
    assert r["level"] == "L2"


def test_survey_still_ignores_g3(client):
    """G3 — источник целевой переменной, в анкете не принимается никогда."""
    s = client.get("/api/schema").json()
    core = {f["name"]: f["default"] for f in s["features"] if f["name"] in s["survey_core"]}
    clean = client.post("/api/predict/single", json={"answers": core}).json()
    dirty = client.post("/api/predict/single",
                        json={"answers": {**core, "G3": 0}}).json()
    assert clean["probability"] == dirty["probability"]


def test_survey_rejects_bad_grades(client):
    s = client.get("/api/schema").json()
    core = {f["name"]: f["default"] for f in s["features"] if f["name"] in s["survey_core"]}
    for bad in ("abc", 99, -3):
        r = client.post("/api/predict/single", json={"answers": {**core, "G1": bad, "G2": 10}})
        assert r.status_code == 422, f"балл {bad!r} должен отклоняться"


def test_survey_accepts_partial_and_reports_defaults(client):
    """Анкета короткая: неспрошенное подставляется типичным значением потока,
    и в ответе честно сказано, сколько полей заполнено за пользователя."""
    r = client.post("/api/predict/single", json={"answers": {"failures": 0}})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["defaults_used"] > 0
    assert 0.0 <= j["probability"] <= 1.0


def test_survey_explains_only_what_was_asked(client):
    """В объяснении не должно быть признаков, которых человек не вводил:
    выдавать подставленное значение за «что повлияло» — дезинформация."""
    s = client.get("/api/schema").json()
    core = {f["name"]: f["default"] for f in s["features"] if f["name"] in s["survey_core"]}
    j = client.post("/api/predict/single", json={"answers": core}).json()
    shown = {f["feature"] for f in j["top_factors"]}
    allowed = set(s["survey_core"]) | set(s["survey_grades"])
    assert shown <= allowed, f"объяснение содержит неспрошенное: {shown - allowed}"


def test_survey_excludes_intrusive_questions(client):
    """Главное в переделке анкеты — не длина, а то, о чём НЕ спрашиваем.

    Алкоголь, романтические отношения, работа и образование родителей, опекун,
    состав семьи — вторжение в личное, которое вдобавок ничего не даёт
    предсказанию (у половины этих признаков важность отрицательная).
    """
    s = client.get("/api/schema").json()
    core = set(s["survey_core"])
    forbidden = {"Dalc", "Walc", "romantic", "Mjob", "Fjob", "Medu", "Fedu",
                 "guardian", "famsize", "Pstatus", "nursery", "reason", "school"}
    assert not (core & forbidden), f"в анкете вернулись лишние вопросы: {core & forbidden}"


def test_survey_is_manageable(client):
    """Анкета короче полного набора признаков и заканчивается необязательным
    шагом с оценками — иначе её просто не дозаполняют."""
    s = client.get("/api/schema").json()
    assert len(s["survey_core"]) < len(s["survey_features"]), "анкета должна быть короче полного набора"
    steps = s["survey_steps_short"]
    assert all(len(x["features"]) <= 8 for x in steps), "шаг не должен быть длинной простынёй"
    assert steps[-1].get("optional"), "последний шаг (оценки) должен быть необязательным"


def test_survey_names_the_subject(client):
    """«Сколько раз вы не сдавали этот предмет» — какой предмет?
    Без ответа на это вопрос повисает в воздухе."""
    s = client.get("/api/schema").json()
    assert s["subject"], "предмет должен быть назван"
    failures = next(f for f in s["features"] if f["name"] == "failures")
    assert s["subject"] in (failures.get("self_label") or ""),         "в вопросе про незачёты должен быть назван предмет"


def test_demo_marks_in_sample(client):
    j = client.post("/api/predict/demo").json()
    assert j["file"]["demo"] is True
    assert j["verification"]["in_sample"] is True
    assert "завышен" in j["verification"]["note"]


def test_demo_uses_main_model_not_saturated_l1(client):
    """Демо показывает L2: L1 на своей обучающей выборке даёт сплошные 100 %."""
    j = client.post("/api/predict/demo").json()
    assert j["level"] == "L2"
    probs = [r["probability"] for r in j["rows"]]
    assert max(probs) < 1.0, "вероятности упёрлись в единицу — демо бесполезно"
    assert len({round(p, 2) for p in probs}) > 20, "распределение вырождено"
