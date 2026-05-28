# Reviews the results.json of an executed analysis and writes a report.
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _load_skill(name: str) -> str:
    """Load a skill body (everything after the closing `---` of the frontmatter)."""
    text = (_SKILLS_DIR / name / "SKILL.md").read_text()
    parts = text.split("---\n", 2)
    return parts[2].strip() if len(parts) == 3 else text


def _build_system() -> str:
    return f"""\
You are an expert reviewer of observational causal inference (OCI) analyses.

You will receive:
1. The pre-analysis plan — a natural-language description of the analysis.
2. The analysis spec (YAML) — treatment, outcome, covariates, estimands, and
   the notebook used to run the analysis.
3. results.json — raw output from the executed analysis notebook. Its
   `estimands` map keys each requested estimand (ate / att / ato) to a dict
   with estimate, stderr, ci_lower, ci_upper, and a per-estimand
   `diagnostics` object containing covariate_balance, overlap, and placebo
   (diagnostics depend on the estimand's weighting and trimming, so they
   are reported separately for each).

Follow these two skills exactly. They are the authoritative source for how to
write the report and how to propose remedies. Where a skill marks a step as
MUST or as a primary/secondary remedy, follow that ordering — do not
substitute your own framing.

=== writing-reports skill ===
{_load_skill("writing-reports")}

=== suggesting-remedies skill ===
{_load_skill("suggesting-remedies")}

Return a JSON code block with:
- "satisfaction": object keyed by the estimands actually estimated in this
  run (subset of {"ate", "att", "ato"} — include only those present in
  results.json). Each value is one of "fully_satisfactory" |
  "satisfactory_with_caveats" | "not_satisfactory", judged independently
  per estimand against its own balance / overlap / placebo diagnostics.
  Apply the writing-reports three-tier rule per estimand: a
  covariate-balance ⛔ on *this estimand's* diagnostics blocks; overlap or
  placebo ⚠️ warnings on *this estimand* demote to
  "satisfactory_with_caveats" but never to "not_satisfactory". Do not
  collapse the three estimands to a single overall verdict — each has
  separate weighting / trimming and so distinct diagnostics.
- "summary": one-paragraph summary of analysis quality and findings.
- "issues": list[str] of specific problems found (empty if all estimands
  are fully_satisfactory).
- "suggestions": list[str] of concrete spec changes for the actor to try.
- "report": full markdown report following the writing-reports structure.
"""


_SYSTEM = _build_system()


SATISFACTION_LEVELS = ("fully_satisfactory", "satisfactory_with_caveats", "not_satisfactory")
ESTIMANDS = ("ate", "att", "ato")


@dataclass
class CritiqueResult:
    satisfaction: dict[str, str]  # {estimand: tier}; only enabled estimands
    summary: str
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    report: str = ""

    @property
    def is_satisfactory(self) -> bool:
        """Derived bool: True iff no enabled estimand is `not_satisfactory`."""
        return all(v != "not_satisfactory" for v in self.satisfaction.values())

    @property
    def worst_satisfaction(self) -> str:
        """Worst tier across enabled estimands — handy for one-line summaries."""
        order = {t: i for i, t in enumerate(SATISFACTION_LEVELS)}
        if not self.satisfaction:
            return "not_satisfactory"
        return max(self.satisfaction.values(), key=lambda t: order.get(t, 0))


class Critic:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        # Anthropic() reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from the
        # environment; we don't supply defaults so the caller controls auth.
        self.client = anthropic.Anthropic()
        self.model = model
        self.last_interaction: dict | None = None

    def evaluate(
        self,
        results: str | dict,
        plan_text: str | None = None,
        spec_text: str | None = None,
        iteration: int | None = None,
    ) -> CritiqueResult:
        if not isinstance(results, str):
            results = json.dumps(results, indent=2, default=str)

        user_content = ""
        if iteration is not None:
            user_content += f"## Iteration\n\nThis is iter_{iteration:02d}.\n\n---\n\n"
        if plan_text:
            user_content += f"## Pre-analysis Plan\n\n{plan_text}\n\n---\n\n"
        if spec_text:
            user_content += f"## Analysis Spec\n\n```yaml\n{spec_text}\n```\n\n---\n\n"
        user_content += f"## results.json\n\n```json\n{results[:40_000]}\n```"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_content}],
        )

        raw_text = response.content[0].text
        self.last_interaction = {
            "system": _SYSTEM[:500] + "..." if len(_SYSTEM) > 500 else _SYSTEM,
            "user_content": user_content[:3000] + "..." if len(user_content) > 3000 else user_content,
            "response": raw_text,
            "model": self.model,
        }

        return _parse_critique(raw_text)


def _parse_critique(text: str) -> CritiqueResult:
    # The report string can contain nested ```yaml / ```python fences, so a
    # non-greedy match would truncate. Find the first ```json opener, then
    # the LAST ``` in the remaining text.
    start = re.search(r"```json\s*", text)
    body_end = text.rfind("```")
    if start and body_end > start.end():
        try:
            data = json.loads(text[start.end():body_end])
            suggestions = data.get("suggestions", [])
            if isinstance(suggestions, str):
                suggestions = [suggestions] if suggestions else []
            satisfaction = _normalize_satisfaction(data)
            return CritiqueResult(
                satisfaction=satisfaction,
                summary=data.get("summary", ""),
                issues=data.get("issues", []),
                suggestions=suggestions,
                report=data.get("report", text),
            )
        except Exception:
            pass
    return CritiqueResult(
        satisfaction={est: "not_satisfactory" for est in ESTIMANDS},
        summary="Could not parse critique response.",
        issues=["Critique parse error"],
        report=text,
    )


def _normalize_satisfaction(data: dict) -> dict[str, str]:
    """Coerce `satisfaction` into {estimand: tier}. Accepts the per-estimand
    dict (new schema) or a single string / legacy `is_satisfactory` boolean
    (old schema, broadcast across all three estimands)."""
    raw = data.get("satisfaction")
    if isinstance(raw, dict):
        result = {}
        for est in ESTIMANDS:
            v = raw.get(est)
            if v in SATISFACTION_LEVELS:
                result[est] = v
        if result:
            return result
    if isinstance(raw, str) and raw in SATISFACTION_LEVELS:
        return {est: raw for est in ESTIMANDS}
    legacy = data.get("is_satisfactory")
    tier = "fully_satisfactory" if legacy else "not_satisfactory"
    return {est: tier for est in ESTIMANDS}
