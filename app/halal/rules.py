"""Rule-based halal checker for Korean (and English) menu text.

Rules are deterministic and run first. The LLM may add details, but it can
never downgrade a HARAM verdict that came from these rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum


class Verdict(IntEnum):
    """Ordered by severity so max() picks the strictest verdict."""
    LIKELY_OK = 0   # no meat/alcohol expected - still not a certification
    DOUBTFUL = 1    # ask the restaurant (meat slaughter, broth, sauces)
    HARAM = 2       # contains pork or alcohol

    @property
    def label(self) -> str:
        return {0: "likely_ok", 1: "doubtful", 2: "haram"}[self.value]


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    verdict: Verdict
    reason_uz: str
    reason_en: str


def _r(regex: str, verdict: Verdict, uz: str, en: str) -> Rule:
    return Rule(re.compile(regex, re.IGNORECASE), verdict, uz, en)


H, D = Verdict.HARAM, Verdict.DOUBTFUL

# Order does not matter: every matching rule is reported.
RULES: list[Rule] = [
    # ---- pork ---------------------------------------------------------
    _r(r"돼지|돈육|돈사태", H, "Cho'chqa go'shti", "Pork"),
    _r(r"삼겹|오겹|목살|항정|가브리", H, "Cho'chqa go'shti (qorin/bo'yin qismi)", "Pork cut"),
    _r(r"제육", H, "Jeyuk — cho'chqa go'shti", "Jeyuk is pork"),
    _r(r"보쌈|족발|순대|수육", H, "Odatda cho'chqa go'shtidan", "Usually pork"),
    _r(r"돈[까가]스|돈부리|탕수육", H, "Odatda cho'chqa go'shtidan", "Usually pork"),
    _r(r"감자탕|뼈해장국|순대국|돼지국밥", H, "Cho'chqa suyagi sho'rvasi", "Pork bone soup"),
    _r(r"베이컨|스팸|소시지|소세지|차슈|햄(?!버거)", H, "Kolbasa/vetchina (cho'chqa)", "Processed pork"),
    _r(r"라드|lard", H, "Cho'chqa yog'i", "Lard"),
    _r(r"\b(pork|bacon|ham|sausage|spam|chashu)\b", H, "Cho'chqa go'shti", "Pork"),
    # ---- alcohol --------------------------------------------------------
    _r(r"소주|맥주|막걸리|청주|사케|와인|위스키|하이볼|칵테일|생맥", H, "Spirtli ichimlik", "Alcoholic drink"),
    _r(r"미림|맛술|술(?!안주)", H, "Tarkibida spirt bor", "Contains alcohol"),
    _r(r"\b(soju|beer|wine|makgeolli|sake|whisk(?:e)?y|highball|cocktail)\b", H, "Spirtli ichimlik", "Alcoholic drink"),
    # ---- doubtful -----------------------------------------------------------
    _r(r"소고기|쇠고기|한우|불고기|(?<![닭돼])갈비|차돌|육회|곰탕|설렁탕|햄버거", D,
       "Mol go'shti — halol so'yilganini so'rang", "Beef - ask if halal-slaughtered"),
    _r(r"닭|치킨|오리", D, "Tovuq/o'rdak — halol so'yilganini so'rang", "Poultry - ask if halal-slaughtered"),
    _r(r"양고기|양꼬치", D, "Qo'y go'shti — halol so'yilganini so'rang", "Lamb - ask if halal-slaughtered"),
    _r(r"육수|사골", D, "Sho'rva asosi noma'lum go'shtdan bo'lishi mumkin", "Broth may be from non-halal meat"),
    _r(r"젤라틴|gelatin", D, "Jelatin manbasini so'rang", "Ask about gelatin source"),
    _r(r"\b(beef|chicken|lamb|duck|broth)\b", D, "Go'sht — halol so'yilganini so'rang", "Meat - ask about slaughter"),
]


@dataclass
class RuleResult:
    verdict: Verdict = Verdict.LIKELY_OK
    reasons_uz: list[str] = field(default_factory=list)
    reasons_en: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)


def check_text(text: str) -> RuleResult:
    """Return the strictest verdict for `text` plus every matching reason."""
    result = RuleResult()
    for rule in RULES:
        m = rule.pattern.search(text)
        if not m:
            continue
        result.verdict = max(result.verdict, rule.verdict)
        if rule.reason_uz not in result.reasons_uz:
            result.reasons_uz.append(rule.reason_uz)
            result.reasons_en.append(rule.reason_en)
        result.matched.append(m.group(0))
    return result
