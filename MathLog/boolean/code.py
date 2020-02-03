# -*- coding: utf-8 -*-

# Скобки
OPEN_P = 1         # (
CLOSE_P = 2        # )

# Конъюнкция
CON_MUL = 10       # *
CON_AMP = 11       # &
CON_CUP = 12       # ^
CON_AND = 13       # AND

# Дизъюнкция
DIS_SUM = 14       # +
DIS_CUP1 = 15      # ∨
DIS_CUP2 = 16      # v - англ. буква
DIS_OR = 17        # OR

# Инверсия
INV_SYMB = 18      # ~
INV_NOT = 19       # NOT

# Импликация
IMP1 = 20          # →
IMP2 = 21          # ⇒

# Строгая дизъюнкиця
STR_DIS = 22       # ⊕
STR_DIS_XOR = 23   # XOR

# Эквивалентность
EQUIV = 24         # ⇔
EQUIV_EQ = 25      # EQ

# Или-Не
NOR = 26           # ↓
NOR_W = 27         # NOR

# И-Не
NAND = 28          # |
NAND_W = 29        # NAND

# Код символа -> строка
NTS = {
    1: "(", 2: ")",
    10: "*", 11: "&", 12: "^", 13: "AND",
    14: "+", 15: "∨", 16: "v", 17: "OR",
    18: "~", 19: "NOT",
    20: "→", 21: "⇒",
    22: "⊕", 23: "XOR",
    24: "⇔", 25: "EQ",
    26: "↓", 27: "NOR",
    28: "|", 29: "NAND",
}

# Символ - код символа
STN = {
    "(": 1, ")": 2,
    "*": 10, "&": 11, "^": 12, "AND": 13,
    "+": 14, "∨": 15, "v": 16, "OR": 17,
    "~": 18, "NOT": 19,
    "→": 20, "⇒": 21,
    "⊕": 22, "XOR": 23,
    "⇔": 24, "EQ": 25,
    "↓": 26, "NOR": 27,
    "|": 28, "NAND": 29,
}

# Код символа -> код операции
NTP = {
    10: "AND", 11: "AND", 12: "AND", 13: "AND",
    14: "OR", 15: "OR", 16: "OR", 17: "OR",
    18: "NOT", 19: "NOT",
    20: "IMPL", 21: "IMPL",
    22: "XOR", 23: "XOR",
    24: "EQ", 25: "EQ",
    26: "NOR", 27: "NOR",
    28: "NAND", 29: "NAND",
}

PRIORITIES = {
    "NOT": 1,
    "AND": 2,
    "NAND": 2,
    "OR": 3,
    "XOR": 4,
    "NOR": 4,
    "IMPL": 5,
    "EQ": 6,
}

VALID_SYMBOLS = " 0123456789()*&^+∨v~→⇒⊕⇔↓|ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
