import pytest

from app.halal.rules import Verdict, check_text


@pytest.mark.parametrize("text", ["삼겹살", "돼지국밥", "제육볶음", "김치찌개 (돼지고기)", "베이컨 피자", "Pork cutlet", "족발 보쌈"])
def test_pork_is_haram(text):
    assert check_text(text).verdict == Verdict.HARAM


@pytest.mark.parametrize("text", ["소주", "생맥주 500cc", "막걸리", "불고기 (미림)", "Highball"])
def test_alcohol_is_haram(text):
    assert check_text(text).verdict == Verdict.HARAM


@pytest.mark.parametrize("text", ["불고기", "닭갈비", "양꼬치", "갈비탕", "Chicken burger", "햄버거"])
def test_meat_is_doubtful(text):
    assert check_text(text).verdict == Verdict.DOUBTFUL


@pytest.mark.parametrize("text", ["떡볶이", "해물파전", "비빔밥", "공기밥", "Seafood pancake"])
def test_no_meat_keyword_is_likely_ok(text):
    assert check_text(text).verdict == Verdict.LIKELY_OK


def test_hamburger_is_not_pork_ham():
    assert check_text("햄버거").verdict == Verdict.DOUBTFUL
    assert check_text("햄").verdict == Verdict.HARAM


def test_strictest_verdict_wins_and_reasons_are_listed():
    r = check_text("닭갈비 + 소주 세트")
    assert r.verdict == Verdict.HARAM
    assert len(r.reasons_uz) == 2
