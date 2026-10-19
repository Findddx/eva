"""Scalar math reference for optional higher precision CPU activation research.

No inference backend uses this module. Inputs are already in the post-input-
Hadamard projection space. Python float arithmetic tests the quantization and
integer-dot identity, not GPU FP16 equivalence or SIMD performance.
"""
import math
import struct

MUL1_MULT = 0x83DCD12D
MUL1_SCALE = struct.unpack("<e", bytes.fromhex("ee1e"))[0]
WIDE_LEVELS = 127 * 128


def split_activation(values, precision="residual8x2"):
    x = list(map(float, values))
    if not x or any(not math.isfinite(value) for value in x):
        raise ValueError("activation must be nonempty and finite")
    amax = max(map(abs, x))
    if precision not in ("int8", "wide15", "residual8x2"):
        raise ValueError("unknown reference activation precision")
    if not amax:
        return [(1., [0] * len(x))]
    if precision == "wide15":
        scale = amax / WIDE_LEVELS
        q = [max(-WIDE_LEVELS, min(WIDE_LEVELS, round(value / scale))) for value in x]
        high = [(value + 64) >> 7 for value in q]
        low = [value - 128 * hi for value, hi in zip(q, high)]
        return [(scale * 128, high), (scale, low)]
    scale = amax / 127
    high = [max(-127, min(127, round(value / scale))) for value in x]
    result = [(scale, high)]
    if precision == "residual8x2":
        residual = [value - scale * hi for value, hi in zip(x, high)]
        residual_max = max(map(abs, residual))
        if residual_max:
            low_scale = residual_max / 127
            low = [max(-127, min(127, round(value / low_scale))) for value in residual]
            result.append((low_scale, low))
    return result


def mul1_integer(state):
    product = (int(state) * MUL1_MULT) & 0xFFFFFFFF
    return sum((product >> bit) & 255 for bit in (0, 8, 16, 24)) - 510


def project_mul1(values, columns, precision=None):
    """Decode each weight once; use one/two integer accumulators per column."""
    rows = None if precision is None else split_activation(values, precision)
    output, decoded = [], 0
    for states in columns:
        if len(states) != len(values):
            raise ValueError("projection input and state count differ")
        coefficients = [mul1_integer(state) for state in states]
        decoded += len(coefficients)
        if rows is None:
            output.append(MUL1_SCALE * math.fsum(x * w for x, w in zip(values, coefficients)))
        else:
            terms = [scale * sum(q * w for q, w in zip(quantized, coefficients))
                     for scale, quantized in rows]
            output.append(MUL1_SCALE * math.fsum(terms))
    return output, decoded


def int32_safe_inputs(weight_bound, activation_bound):
    return ((1 << 31) - 1) // (int(weight_bound) * int(activation_bound))
