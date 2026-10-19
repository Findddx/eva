"""Bounded, prompt-only literal-default edit candidates; no production caller.

This parses a deliberately narrow English instruction and an independently
provided UTF-8 module. It does not read files, model output, token history,
expected answers or quality assertions. A returned proposal still requires
tokenizer alignment and exact target verification before any token is emitted.
"""
from dataclasses import dataclass
import ast
import hashlib
import io
import keyword
import math
import re
import tokenize


MAX_SOURCE_BYTES = 1024 * 1024
MAX_INSTRUCTION_CHARS = 8192
MAX_LITERAL_CHARS = 4096
MAX_LITERAL_NODES = 1024
MAX_LITERAL_DEPTH = 32

_CHANGE = re.compile(
    r"\bChange(?: only)? the default (?P<parameter>\S+) parameter of "
    r"(?P<function>\S+) from ", re.IGNORECASE)
_LEADING = re.compile(
    r"Update the following Python module(?: used by [^.!?\r\n]{1,240})?\.",
    re.IGNORECASE)
_OTHER_EDITS = re.compile(r"\b(change|delete|remove|rename|add|insert|replace|alter|move|reorder)\b", re.IGNORECASE)
_FRAMES = (
    re.compile(r"\s+Return the complete updated module as Python code\.$", re.IGNORECASE),
    re.compile(r"\s+Preserve all other code(?:, comments)? and (?:docstrings|comments)\.$", re.IGNORECASE),
)


@dataclass(frozen=True)
class EditProposal:
    """One source splice, not a completion or a validated target answer."""
    proposed_source: bytes
    function: str
    parameter: str
    old_literal: str
    new_literal: str
    start_byte: int
    end_byte: int
    replacement_utf8: bytes
    source_sha256: str
    instruction_sha256: str
    proposal_sha256: str
    candidate_only: bool = True


def _identifier(text, *, function=False):
    if text.startswith("`") and text.endswith("`"):
        text = text[1:-1]
    if function and text.endswith("()"):
        text = text[:-2]
    return text if text.isidentifier() and not keyword.iskeyword(text) else None


def _unquote_literal(text):
    text = text.strip()
    if text.startswith("`") and text.endswith("`"):
        text = text[1:-1].strip()
    return text


def _literal(text):
    """Literal grammar only; even literal_eval's special set() is excluded."""
    if not text or len(text) > MAX_LITERAL_CHARS:
        return None
    try:
        node = ast.parse(text, mode="eval").body
        stack, count = [(node, 0)], 0
        while stack:
            value, depth = stack.pop()
            count += 1
            if count > MAX_LITERAL_NODES or depth > MAX_LITERAL_DEPTH:
                return None
            if isinstance(value, ast.Constant):
                if type(value.value) not in (type(None), bool, int, float, complex, str, bytes):
                    return None
                if isinstance(value.value, (float, complex)):
                    if not (math.isfinite(value.value.real) and math.isfinite(value.value.imag)):
                        return None
            elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                stack.extend((child, depth+1) for child in value.elts)
            elif isinstance(value, ast.Dict):
                if any(key is None for key in value.keys):
                    return None
                stack.extend((child, depth+1) for child in (*value.keys, *value.values))
            elif isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
                if not isinstance(value.operand, ast.Constant) or type(value.operand.value) not in (int, float, complex):
                    return None
                stack.append((value.operand, depth+1))
            else:
                return None
        result = ast.literal_eval(node)
        return node, _fingerprint(result)
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError, MemoryError):
        return None


def _fingerprint(value):
    """Type and zero-sign sensitive; 1, True and 1.0 are different defaults."""
    kind = type(value).__name__
    if isinstance(value, float):
        return kind, value.hex()
    if isinstance(value, complex):
        return kind, value.real.hex(), value.imag.hex()
    if isinstance(value, (tuple, list)):
        return kind, tuple(_fingerprint(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return kind, tuple(sorted((_fingerprint(item) for item in value), key=repr))
    if isinstance(value, dict):
        return kind, tuple((_fingerprint(key), _fingerprint(item)) for key, item in value.items())
    return kind, value


def _instruction(instruction):
    if not isinstance(instruction, str) or not 0 < len(instruction) <= MAX_INSTRUCTION_CHARS:
        return None
    matches = list(_CHANGE.finditer(instruction))
    if len(matches) != 1:
        return None
    match = matches[0]
    leading = instruction[:match.start()].strip()
    if leading and (not _LEADING.fullmatch(leading) or _OTHER_EDITS.search(leading)):
        return None
    parameter = _identifier(match["parameter"])
    function = _identifier(match["function"], function=True)
    if function is None or parameter is None:
        return None
    tail = instruction[match.end():].strip()
    # These optional, generic framing sentences do not introduce edits. Any
    # unknown trailing prose remains and fails the literal parser below.
    for frame in _FRAMES:
        tail = frame.sub("", tail).rstrip()
    if tail.endswith("."):
        tail = tail[:-1].rstrip()  # English terminator; write float4.0 or `4.`.
    splits = []
    for separator in re.finditer(r" to ", tail, re.IGNORECASE):
        old = _unquote_literal(tail[:separator.start()])
        new = _unquote_literal(tail[separator.end():])
        parsed_old, parsed_new = _literal(old), _literal(new)
        if parsed_old is not None and parsed_new is not None:
            splits.append((old, new, parsed_old[1], parsed_new[1]))
    if len(splits) != 1:
        return None
    return function, parameter, *splits[0]


def _default(tree, function, parameter):
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function]
    if len(definitions) != 1 or definitions[0] not in tree.body:
        return None  # No class/method/nested/overload disambiguation by guessing.
    args = definitions[0].args
    positional = [*args.posonlyargs, *args.args]
    all_args = [*positional, *args.kwonlyargs]
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    if len({arg.arg for arg in all_args}) != len(all_args):
        return None
    defaults = dict(zip((arg.arg for arg in positional[len(positional)-len(args.defaults):]), args.defaults))
    defaults.update((arg.arg, value) for arg, value in zip(args.kwonlyargs, args.kw_defaults) if value is not None)
    return defaults.get(parameter)


def _source(source):
    if isinstance(source, str):
        try:
            source = source.encode("utf-8")
        except UnicodeError:
            return None
    if not isinstance(source, bytes) or not 0 < len(source) <= MAX_SOURCE_BYTES:
        return None
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        if encoding not in ("utf-8", "utf-8-sig"):
            return None
        bom = 3 if source.startswith(b"\xef\xbb\xbf") else 0
        body = source[bom:]
        text = body.decode("utf-8")
        return source, body, bom, ast.parse(text)
    except (SyntaxError, UnicodeError, ValueError, RecursionError, MemoryError):
        return None


def propose_default_edit(instruction, source_utf8):
    """Return a unique prompt-only candidate, or None on unsupported input.

    Supported instruction: "Change [only] the default PARAM parameter of FUNC
    from OLD_LITERAL to NEW_LITERAL." Optional module/preserve/return framing
    is documented. Function/parameter identifiers and literals come from that
    instruction, never a fixed task. No source is imported or executed.
    """
    request, source = _instruction(instruction), _source(source_utf8)
    if request is None or source is None:
        return None
    function, parameter, old, new, old_value, new_value = request
    original, body, bom, tree = source
    target = _default(tree, function, parameter)
    if target is None or target.end_lineno is None or target.end_col_offset is None:
        return None
    lines = body.splitlines(keepends=True)
    if not (1 <= target.lineno <= target.end_lineno <= len(lines)):
        return None
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    start = bom+offsets[target.lineno-1]+target.col_offset
    end = bom+offsets[target.end_lineno-1]+target.end_col_offset
    if not 0 <= start < end <= len(original):
        return None
    try:
        literal_source = original[start:end].decode("utf-8")
        actual = _literal(literal_source)
        if actual is None or actual[1] != old_value:
            return None
        if any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(literal_source).readline)):
            return None  # Replacing a compound literal must not drop its comments.
        replacement = new.encode("utf-8")
        proposed = original[:start]+replacement+original[end:]
        checked = _source(proposed)
        if checked is None:
            return None
        new_default = _default(checked[3], function, parameter)
        if new_default is None:
            return None
        # Recheck the resulting AST literal without compiling/executing code.
        value = _literal(ast.unparse(new_default))
        if value is None or value[1] != new_value:
            return None
        return EditProposal(proposed, function, parameter, old, new, start, end, replacement,
            hashlib.sha256(original).hexdigest(), hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            hashlib.sha256(proposed).hexdigest())
    except (SyntaxError, UnicodeError, ValueError, TypeError, tokenize.TokenError, RecursionError, MemoryError):
        return None
