import math


def passes_quality_gate(r2: float, minimum_r2: float) -> bool:
    return math.isfinite(r2) and r2 >= minimum_r2
