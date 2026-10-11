import re

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str, n_sources: int) -> list[int]:
    """Source numbers cited in `text`, in order of first appearance. Unknown numbers are dropped."""
    seen: list[int] = []
    for m in CITE.finditer(text):
        for part in m.group(1).split(","):
            n = int(part)
            if 1 <= n <= n_sources and n not in seen:
                seen.append(n)
    return seen


def strip_invalid(text: str, n_sources: int) -> str:
    """Removes citation markers that point at sources that don't exist."""

    def fix(m: re.Match) -> str:
        valid = [p.strip() for p in m.group(1).split(",") if 1 <= int(p) <= n_sources]
        return f"[{', '.join(valid)}]" if valid else ""

    return CITE.sub(fix, text)
