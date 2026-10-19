"""Confidence-and-cold-byte gate for DSpark drafts.

Truncating a draft changes how many tokens are verified together. It does not
change the distribution of tokens that are accepted.
"""


def truncate_draft(confidences, cold_bytes, conf_min: float, byte_budget: int) -> int:
    depth = 0
    used = 0
    for conf, nbytes in zip(confidences, cold_bytes):
        if conf < conf_min or used + nbytes > byte_budget:
            break
        used += int(nbytes)
        depth += 1
    return depth
