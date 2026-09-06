import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import joblib
import numpy as np

# --- Определяем пути ---
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
    EXTERNAL_DIR = Path(sys.executable).parent
    sys.path.append(str(BASE_DIR / "src"))
else:
    BASE_DIR = Path(__file__).parent
    EXTERNAL_DIR = BASE_DIR
    sys.path.append(str(BASE_DIR / "src"))

from scoring import fit_blend, predict_risk

BEST_RF = {"max_features": 0.3, "min_samples_leaf": 10, "max_depth": None}
BLEND_W = (0.4, 0.3, 0.3)
DEFAULT_K = 130

EMBEDDED_MODEL_PATH = BASE_DIR / "models" / "blend.joblib"

RU = {
    "school": "Учебное заведение",
    "failures": "Прошлые незачёты",
    "higher": "Планирует учиться дальше",
    "guardian": "Опекун",
    "absences": "Пропуски занятий",
    "studytime": "Часы самоподготовки",
}

def translate_factor(factor):
    base = factor.split("_")[0] if "_" in factor else factor
    return RU.get(base, base)

def load_embedded_model():
    if not EMBEDDED_MODEL_PATH.exists():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Ошибка", "Встроенная модель не найдена.")
        sys.exit(1)
    return joblib.load(EMBEDDED_MODEL_PATH)

model = load_embedded_model()
model_source = "встроенная"

# ---------- НАСТРОЙКА ТЁМНОЙ ТЕМЫ ----------
BG = "#1e1e1e"
PANEL_BG = "#2b2b2b"
BUTTON_BG = "#3c3c3c"
BUTTON_HOVER = "#4a4a4a"
BUTTON_ACTIVE = "#555555"
TEXT = "#ffffff"
TEXT_DIM = "#bbbbbb"
ENTRY_BG = "#333333"
TABLE_BG = "#252526"
TABLE_HEADER_BG = "#2d2d30"
TABLE_SELECT_BG = "#094771"

root = tk.Tk()
root.title("Сервис ранжирования студентов по риску незачёта")
root.geometry("1000x650")
root.minsize(800, 500)
root.configure(bg=BG)

style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview",
                background=TABLE_BG,
                foreground=TEXT,
                fieldbackground=TABLE_BG,
                borderwidth=0,
                font=("Segoe UI", 10))
style.configure("Treeview.Heading",
                background=TABLE_HEADER_BG,
                foreground=TEXT,
                font=("Segoe UI", 10, "bold"),
                relief="flat")
style.map("Treeview",
          background=[("selected", TABLE_SELECT_BG)],
          foreground=[("selected", TEXT)])
style.configure("Vertical.TScrollbar",
                background=BUTTON_BG,
                troughcolor=BG,
                bordercolor=BG,
                arrowcolor=TEXT)

# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ КНОПОК С ХОВЕРОМ ----------
def create_button(parent, text, command, width=20, height=2, font=("Segoe UI", 10, "bold")):
    btn = tk.Button(parent, text=text, command=command,
                    width=width, height=height,
                    bg=BUTTON_BG, fg=TEXT,
                    activebackground=BUTTON_ACTIVE, activeforeground=TEXT,
                    relief="flat", bd=0, cursor="hand2", font=font)
    btn.bind("<Enter>", lambda e: btn.config(bg=BUTTON_HOVER))
    btn.bind("<Leave>", lambda e: btn.config(bg=BUTTON_BG))
    return btn

# ---------- КОМПОНОВКА ----------
# Верхняя статусная панель
status_frame = tk.Frame(root, bg=BG)
status_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
status_label = tk.Label(status_frame, text=f"Текущая модель: {model_source}",
                        font=("Segoe UI", 9, "bold"), bg=BG, fg=TEXT_DIM)
status_label.pack(anchor=tk.W)

# Заголовок
title_label = tk.Label(root, text="Ранжирование студентов по риску незачёта",
                       font=("Segoe UI", 14, "bold"), bg=BG, fg=TEXT)
title_label.pack(pady=(10, 5))

# Инструкция
instr_label = tk.Label(root, text="Загрузите CSV-файл со студентами (разделитель ';').\n"
                                   "Будут показаны студенты с наибольшим риском (по умолчанию топ-130).",
                       font=("Segoe UI", 10), bg=BG, fg=TEXT_DIM, justify=tk.LEFT)
instr_label.pack(pady=(0, 10))

# Панель с выбором K
filter_frame = tk.Frame(root, bg=BG)
filter_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(filter_frame, text="Сколько студентов показать (K):", bg=BG, fg=TEXT).pack(side=tk.LEFT)
k_var = tk.IntVar(value=DEFAULT_K)
k_entry = tk.Entry(filter_frame, textvariable=k_var, width=8,
                   font=("Segoe UI", 10), bg=ENTRY_BG, fg=TEXT,
                   insertbackground=TEXT, relief="flat", bd=0)
k_entry.pack(side=tk.LEFT, padx=5)

apply_k_btn = create_button(filter_frame, "Показать топ-K", lambda: apply_k(), width=15, height=1)
apply_k_btn.pack(side=tk.LEFT, padx=5)

show_all_btn = create_button(filter_frame, "Показать всех", lambda: show_all(), width=15, height=1)
show_all_btn.pack(side=tk.LEFT, padx=5)

# Таблица
table_frame = tk.Frame(root, bg=BG)
table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

columns = ("Место", "Школа", "Прошлые незачёты", "Хочет учиться дальше", "Пропуски", "Риск", "Пол", "Возраст")
tree = ttk.Treeview(table_frame, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=90 if col != "Место" else 60,
                anchor=tk.CENTER if col in ("Место", "Прошлые незачёты", "Пропуски", "Возраст") else tk.W)

scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Подсветка строк по риску
tree.tag_configure("high", background="#3a1f1f")
tree.tag_configure("mid", background="#3a3520")
tree.tag_configure("low", background="#1f3a1f")

loaded_df = None

def display_df(df, top_k=None):
    global loaded_df
    loaded_df = df
    df_sorted = df.sort_values("риск", ascending=False).reset_index(drop=True)
    if top_k is not None:
        df_sorted = df_sorted.head(top_k)

    for row in tree.get_children():
        tree.delete(row)

    for i, r in df_sorted.iterrows():
        risk_val = float(r["риск"])
        if risk_val >= 0.8:
            tag = "high"
        elif risk_val >= 0.5:
            tag = "mid"
        else:
            tag = "low"
        tree.insert("", "end", values=(
            i+1,
            r["school"],
            r["failures"],
            r["higher"],
            r["absences"],
            f"{risk_val:.3f}",
            r["sex"],
            r["age"]
        ), tags=(tag,))

def apply_k():
    try:
        k = k_var.get()
        if loaded_df is not None:
            display_df(loaded_df, top_k=k)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Неверное значение K:\n{e}")

def show_all():
    if loaded_df is not None:
        display_df(loaded_df, top_k=None)

def load_csv():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return
    try:
        df = pd.read_csv(filepath, sep=";")
        risk = predict_risk(df, model)
        df["риск"] = risk
        k = k_var.get()
        display_df(df, top_k=k)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось обработать файл:\n{e}")

def learn_model():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return
    try:
        df = pd.read_csv(filepath, sep=";")
        if "G3" not in df.columns:
            messagebox.showerror("Ошибка", "В файле нет колонки G3 — обучение невозможно.")
            return
        confirm = messagebox.askyesno("Обучение", "Обучение модели займёт несколько секунд.\nПродолжить?")
        if not confirm:
            return
        global model, model_source
        model = fit_blend(df, weights=BLEND_W, random_state=42, rf_params=BEST_RF)
        model["top_factors"] = ["school", "failures", "higher"]
        model_source = "пользовательская (в памяти)"
        status_label.config(text=f"Текущая модель: {model_source}")
        messagebox.showinfo("Готово", "Модель обучена и используется.\nОна не сохранена на диск и исчезнет после закрытия.")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось обучить модель:\n{e}")

def reset_to_embedded():
    global model, model_source
    model = load_embedded_model()
    model_source = "встроенная"
    status_label.config(text=f"Текущая модель: {model_source}")
    messagebox.showinfo("Сброс", "Вернулась встроенная модель.")

def show_factors():
    top_factors = model.get("top_factors", ["school", "failures", "higher"])
    labels = [translate_factor(f) for f in top_factors]
    msg = "Три главных фактора (по текущей модели):\n\n"
    for i, label in enumerate(labels, 1):
        msg += f"{i}. {label}\n"
    msg += "\nЭти признаки сильнее всего влияют на предсказание риска.\nПодробные количественные оценки приведены в аналитическом отчёте."
    messagebox.showinfo("Три фактора", msg)

# Нижние кнопки
btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(fill=tk.X, padx=10, pady=10)

btn_load = create_button(btn_frame, "Применить модель к CSV", load_csv, width=25)
btn_load.pack(side=tk.LEFT, padx=5)

btn_learn = create_button(btn_frame, "Обучить модель на новом CSV", learn_model, width=30)
btn_learn.pack(side=tk.LEFT, padx=5)

btn_factors = create_button(btn_frame, "Показать три фактора", show_factors, width=20)
btn_factors.pack(side=tk.LEFT, padx=5)

btn_reset = create_button(btn_frame, "Сбросить на встроенную", reset_to_embedded, width=20)
btn_reset.pack(side=tk.LEFT, padx=5)

root.mainloop()