import re

from sympy import simplify, sympify

BOXED = "\\boxed"

GOLD_LETTER = re.compile(r"^\(?([A-J])[\):.]")
RESPONSE_LETTER = re.compile(
    r"(?i)answer\s*(?:is)?\s*[:\-]?\s*\**\(?([A-J])\)?(?![A-Za-z])")
RESPONSE_TAIL = re.compile(r"(?i)(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*(.+)")
NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")

UNICODE_MATH = [
    ("\u221a", "\\sqrt"),
    ("\u00d7", "*"),
    ("\u00f7", "/"),
    ("\u2212", "-"),
    ("\u03c0", "pi"),
]

LATEX_PATTERNS = [
    (re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}"), r"((\1)/(\2))"),
    (re.compile(r"\\sqrt\s*\{([^{}]*)\}"), r"sqrt(\1)"),
    (re.compile(r"\\sqrt\s*(\w+)"), r"sqrt(\1)"),
    (re.compile(r"\\text\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\^\s*\{([^{}]*)\}"), r"**(\1)"),
]

LATEX_NOISE = ["\\left", "\\right", "\\!", "\\,", "\\;", "$", "%", "\\"]


def extract_boxed(text: str):
    start = text.rfind(BOXED)
    if start == -1:
        return None

    opening = text.find("{", start)
    if opening == -1:
        return None

    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth = depth + 1
        elif text[position] == "}":
            depth = depth - 1
            if depth == 0:
                return text[opening + 1:position]
    return None


def extract_golden(golden: str, family: str):
    if family == "maths":
        return extract_boxed(golden) or golden

    if family == "multiple_choice":
        found = GOLD_LETTER.match(golden.strip())
        return found.group(1) if found else golden

    return golden


def extract_response(response: str, family: str):
    if not response or not response.strip():
        return None

    boxed = extract_boxed(response)
    if boxed is not None:
        return boxed

    if family == "multiple_choice":
        found = RESPONSE_LETTER.search(response)
        if found:
            return found.group(1)

        tail = RESPONSE_TAIL.search(response)
        if tail:
            letter = GOLD_LETTER.match(tail.group(1).strip())
            return letter.group(1) if letter else None
        return None

    tail = RESPONSE_TAIL.search(response)
    if tail:
        return tail.group(1).strip().split("\n")[0]

    if family == "maths":
        return extract_last_number(response)

    return None


def extract_last_number(text: str):
    found = NUMBER.findall(text[-300:])
    return found[-1] if found else None


def normalise(text) -> str:
    if text is None:
        return ""

    text = str(text).strip().lower()
    for symbol, replacement in UNICODE_MATH:
        text = text.replace(symbol, replacement)
    for pattern, replacement in LATEX_PATTERNS:
        text = pattern.sub(replacement, text)
    for token in LATEX_NOISE:
        text = text.replace(token, "")

    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.strip(".:;")


def to_number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def to_sympy(text):
    try:
        return sympify(text)
    except Exception:
        return None


def answers_match(given, expected) -> bool:
    left = normalise(given)
    right = normalise(expected)

    if left == right:
        return True

    first, second = to_number(left), to_number(right)
    if first is not None and second is not None:
        return abs(first - second) < 1e-6

    first, second = to_sympy(left), to_sympy(right)
    if first is None or second is None:
        return False

    try:
        return simplify(first - second) == 0
    except Exception:
        return False


LONG_ANSWER = 25
TAIL_WINDOW = 400


def contains_answer(response: str, expected: str) -> bool:
    if not response or not expected:
        return False

    window = response if len(expected) >= LONG_ANSWER else response[-TAIL_WINDOW:]
    return normalise(expected) in normalise(window)


def grade(response: str, golden: str, family: str) -> dict:
    expected = extract_golden(golden, family)
    given = extract_response(response, family)

    if given is None and family in ("reasoning", "science"):
        if not response or not response.strip():
            return {"correct": False, "parsed": False,
                    "extracted": None, "expected": expected}

        found = contains_answer(response, expected)
        return {"correct": found, "parsed": True,
                "extracted": expected if found else None,
                "expected": expected}

    if given is None:
        return {"correct": False, "parsed": False,
                "extracted": None, "expected": expected}

    correct = answers_match(given, expected)
    if not correct and family in ("reasoning", "science"):
        correct = contains_answer(response, expected)

    return {"correct": correct, "parsed": True,
            "extracted": given, "expected": expected}
