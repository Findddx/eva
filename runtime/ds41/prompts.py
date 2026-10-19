"""Single-user text prompts matching DeepSeek's official 1017 encoder.

Reference: deepseek-ai/DeepSeek-V4.1-Flash-1017, encoding/encoding.py,
commit 1bae0e1088993b9b82de08f59de5be5a34e42170. No transformers/Jinja.
Tools, images and conversation histories need the full official encoder.
"""

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
SYSTEM = "<｜System｜>"
THINK_START = "<think>"
THINK_END = "</think>"
OFFICIAL_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
ENCODING_REFERENCE = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/"
    + OFFICIAL_COMMIT + "/encoding/encoding.py"
)


def reasoning_budget(effort=None):
    """Official aliases: low=50, high/default=75, max=100; int 1..100."""
    if effort is None:
        return 75
    if type(effort) is int and 1 <= effort <= 100:
        return effort
    if type(effort) is str and effort in {"low", "high", "max"}:
        return {"low": 50, "high": 75, "max": 100}[effort]
    raise ValueError("reasoning_effort must be an integer 1..100 or low/high/max")


def _parts(thinking=False, reasoning_effort=None):
    if type(thinking) is not bool:
        raise ValueError("thinking must be bool")
    budget = reasoning_budget(reasoning_effort)
    prefix = BOS
    if thinking:
        prefix += (SYSTEM + f"Reasoning Effort: {budget} "
                   "(range 1-100, the higher the value, the more thorough the reasoning)\n\n")
    return prefix + USER, ASSISTANT + (THINK_START if thinking else THINK_END)


def render_prompt(prompt, format="chat", thinking=False, reasoning_effort=None):
    """Render one user turn; completion returns the original raw text.

    Chat ends with an open Assistant header plus <think> or </think>. There is
    no EOS before generation. No hidden default system instruction is inserted.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be text")
    if format == "completion":
        if thinking or reasoning_effort is not None:
            raise ValueError("thinking controls apply to chat prompts")
        return prompt
    if format != "chat":
        raise ValueError("format must be chat or completion")
    prefix, suffix = _parts(thinking, reasoning_effort)
    return prefix + prompt + suffix


def encode_prompt(tokenizer, prompt, format="chat", thinking=False,
                  reasoning_effort=None, max_user_tokens=None):
    """Return token IDs using tokenizers.Tokenizer's lightweight API.

    Chat supplies its own BOS and disables tokenizer post-processing. A positive
    max_user_tokens truncates ONLY user body IDs, keeping the role/think suffix.
    None or zero means unlimited. Completion retains the legacy encode() default
    special-token policy; explicitly provided BOS disables a second auto-BOS.
    """
    if max_user_tokens is not None and (type(max_user_tokens) is not int or max_user_tokens < 0):
        raise ValueError("max_user_tokens must be a nonnegative integer or None")
    rendered = render_prompt(prompt, format, thinking, reasoning_effort)
    if format == "completion":
        ids = tokenizer.encode(rendered, add_special_tokens=not prompt.startswith(BOS)).ids
        return ids[:max_user_tokens] if max_user_tokens else ids
    controls = [BOS, USER, ASSISTANT, THINK_START if thinking else THINK_END]
    if thinking:
        controls.append(SYSTEM)
    missing = [s for s in controls if tokenizer.token_to_id(s) is None]
    if missing:
        raise ValueError("Tokenizer lacks required DeepSeek chat tokens: " + ", ".join(missing))
    if not max_user_tokens:
        return tokenizer.encode(rendered, add_special_tokens=False).ids
    body = tokenizer.encode(prompt, add_special_tokens=False).ids
    if len(body) <= max_user_tokens:
        return tokenizer.encode(rendered, add_special_tokens=False).ids
    prefix, suffix = _parts(thinking, reasoning_effort)
    # Preserve exact body token prefix, including incomplete UTF-8 pieces;
    # decoding a truncated body and re-encoding would change the token sequence.
    return (tokenizer.encode(prefix, add_special_tokens=False).ids
            + body[:max_user_tokens]
            + tokenizer.encode(suffix, add_special_tokens=False).ids)
