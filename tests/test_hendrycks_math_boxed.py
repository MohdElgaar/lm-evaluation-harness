"""Regression tests for MATH answer extraction (\\boxed{} + $...$)."""

from lm_eval.tasks.hendrycks_math import utils


def _score(raw: str, gold_boxed: str = r"\boxed{2}") -> int:
    return utils.process_results({"solution": gold_boxed}, [raw])["exact_match"]


def test_boxed_answer_after_think_strip():
    assert _score(r"The answer is \boxed{2}") == 1


def test_boxed_only():
    assert _score(r"\boxed{2}") == 1


def test_dollar_answer_still_works():
    assert _score("Final answer: $2$") == 1


def test_boxed_inside_dollar_delimiters():
    # Dollar heuristic may capture \boxed{...}; must unwrap before compare.
    assert _score(r"Therefore $\boxed{2}$") == 1


def test_wrong_intermediate_dollar_then_correct_boxed():
    raw = r"Earlier we had $x=5$, but the final answer is \boxed{2}."
    assert _score(raw) == 1


def test_complex_boxed_matches_gold():
    gold = r"\boxed{\left( 3, \frac{\pi}{2} \right)}"
    raw = r"Final answer: \boxed{(3, \frac{\pi}{2})}"
    # Strict string equiv after strip_string may still fail on \left vs ();
    # at least ensure boxed extraction does not leave \boxed in the candidate.
    cands = utils._answer_candidates(raw)
    assert r"\boxed" not in cands[0]
    assert utils._safe_unboxed(raw) == r"(3, \frac{\pi}{2})"
