import re
from typing import Dict, List, Union


def _dollar_delimited_answer_candidates(text: str) -> List[str]:
    """Extract math answer candidates from LaTeX delimiters.

    Uses the span between the second-to-last and last single ``$``, and the last
    non-empty ``$$ ... $$`` display block when ``$$`` is used.
    """
    out: List[str] = []
    dollar_pos = [i for i, ch in enumerate(text) if ch == "$"]
    if len(dollar_pos) >= 2:
        span = text[dollar_pos[-2] + 1 : dollar_pos[-1]]
        if span.strip():
            out.append(span)

    if "$$" in text:
        last_block = None
        for m in re.finditer(r"\$\$(.+?)\$\$", text, flags=re.DOTALL):
            inner = m.group(1).strip()
            if inner:
                last_block = inner
        if last_block is not None:
            out.append(last_block)

    return out


def _dedupe_preserve(xs: List[str]) -> List[str]:
    return list(dict.fromkeys(x for x in xs if x is not None and str(x).strip()))


def _strip_rhs_fences(rhs: str) -> str:
    """Trim trailing punctuation and stray `$` models often emit after ``= …``."""
    s = rhs.strip()
    while s and s[-1] in ".,;!?":
        s = s[:-1].strip()
    while s.endswith("$"):
        s = s[:-1].strip()
    while s.startswith("$"):
        s = s[1:].strip()
    return s.strip()


def _equals_rhs_candidates_from(parts: List[str]) -> List[str]:
    """Add RHS of the last `=` in strings that contain one (chains like ``a = b = 277``).

    Coaches/Gemma often omit ``\\boxed{}`` and wrap the verdict in `$… = \\cdots = 077$`; the penultimate/`$$`
    span then still contains the full chain, which does not normalize to bare integer gold.
    """
    out: List[str] = []
    for t in parts:
        if not t or "=" not in t:
            continue
        rhs = _strip_rhs_fences(t.rsplit("=", 1)[-1])
        if rhs:
            out.append(rhs)
    return out


def _exact_match_one(doc: dict, response: str) -> int:
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])

    candidates: List[str] = []

    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None and str(boxed_content).strip():
                candidates.append(boxed_content)
        except (AssertionError, IndexError):
            pass

    candidates.extend(_dollar_delimited_answer_candidates(response))
    candidates.append(response)
    candidates.extend(_equals_rhs_candidates_from(candidates))

    for answer in _dedupe_preserve(candidates):
        if is_equiv(answer, target):
            return 1
    return 0


def process_results(
    doc: dict, results: List[Union[str, List[str]]]
) -> Dict[str, float]:
    """If the default filter keeps K repeats (list of strings), score each and return the mean (avg@K)."""
    first = results[0]
    if isinstance(first, list):
        if not first:
            return {"exact_match": 0.0}
        scores = [_exact_match_one(doc, r) for r in first]
        return {"exact_match": sum(scores) / len(scores)}
    return {"exact_match": float(_exact_match_one(doc, first))}


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
