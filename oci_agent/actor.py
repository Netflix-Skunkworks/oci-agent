# Drafts an analysis spec from a plan, validates it, and revises it from critic suggestions.
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import yaml


_DRAFT_SYSTEM = """\
You are an expert observational causal inference (OCI) analyst.

You will receive a pre-analysis plan (markdown). Invoke the writing-specs skill
to draft an analysis spec.

Return a YAML code block containing the spec, followed by a brief notes
paragraph. The spec must include `analysis_notebook` pointing at an .ipynb
file in the notebooks/ directory. For ACIC evals use the eval field set
(acic_treatment, acic_response, ...); for general analyses use data_path,
treatment, outcome.

YAML key conventions:
- All top-level spec keys MUST be lowercase: `covariates`, `estimate_ate`,
  `estimate_att`, `estimate_ato`, `ato_threshold`, `augment_continuous_covariates`,
  `seed`, `acic_treatment`, `acic_response`, `analysis_notebook`, etc.
- Plans and prose may refer to the corresponding Python variables in UPPERCASE
  (e.g. "set COVARIATES to all x_* columns") because the runner uppercases each
  spec key when injecting it into the notebook's configuration cell. Do NOT
  copy that uppercase into the YAML keys themselves.
"""

_REVISE_SYSTEM = """\
You are an expert observational causal inference (OCI) analyst.

You will receive the previous analysis spec (YAML), the critic's report, and
the critic's suggested spec changes. Invoke the writing-specs skill to apply
the suggestions and produce a revised spec.

Return a YAML code block containing the new spec, followed by a brief notes
paragraph explaining what changed and why.
"""


@dataclass
class ActorResult:
    spec: dict[str, Any]
    notes: str = ""


class Actor:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        # Anthropic() reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from the
        # environment; we don't supply defaults so the caller controls auth.
        self.client = anthropic.Anthropic()
        self.model = model
        self.last_interaction: dict | None = None

    def draft(self, plan_text: str) -> ActorResult:
        return self._call(_DRAFT_SYSTEM, f"## Pre-analysis Plan\n\n{plan_text}")

    def revise(self, prev_spec_text: str, critic_report: str, critic_suggestions: list[str]) -> ActorResult:
        suggestions_md = "\n".join(f"- {s}" for s in critic_suggestions) or "(none)"
        user_content = (
            f"## Previous Spec\n\n```yaml\n{prev_spec_text}\n```\n\n---\n\n"
            f"## Critic Suggestions\n\n{suggestions_md}\n\n---\n\n"
            f"## Critic Report\n\n{critic_report[:20_000]}"
        )
        return self._call(_REVISE_SYSTEM, user_content)

    def propose(self, spec: dict[str, Any]) -> ActorResult:
        """Programmatic validation against the writing-specs Validation rules."""
        covariates = spec.get("covariates") or []
        if not covariates:
            raise ValueError("Validation failed: `covariates` is empty.")
        treatment = spec.get("treatment")
        outcome = spec.get("outcome")
        if treatment and treatment in covariates:
            raise ValueError(f"Validation failed: treatment `{treatment}` listed in covariates.")
        if outcome and outcome in covariates:
            raise ValueError(f"Validation failed: outcome `{outcome}` listed in covariates.")
        if not any(spec.get(k) for k in ("estimate_ate", "estimate_att", "estimate_ato")):
            raise ValueError("Validation failed: at least one estimand toggle must be true.")
        ato = spec.get("ato_threshold")
        if ato is not None and not (0 < ato < 0.5):
            raise ValueError(f"Validation failed: ato_threshold {ato} not in (0, 0.5).")
        nb = spec.get("analysis_notebook")
        if not nb:
            raise ValueError("Validation failed: `analysis_notebook` missing.")
        if not Path(nb).exists():
            raise ValueError(f"Validation failed: notebook not found: {nb}")
        return ActorResult(spec=spec, notes=f"Validated spec ({len(covariates)} covariates, notebook {nb}).")

    def _call(self, system: str, user_content: str) -> ActorResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text
        self.last_interaction = {
            "system": system[:500] + "..." if len(system) > 500 else system,
            "user_content": user_content[:3000] + "..." if len(user_content) > 3000 else user_content,
            "response": raw,
            "model": self.model,
        }
        return _parse_actor_response(raw)


def _parse_actor_response(text: str) -> ActorResult:
    m = re.search(r"```yaml\s*(.*?)\s*```", text, re.DOTALL)
    if not m:
        raise ValueError("Actor response did not contain a YAML code block.")
    spec = yaml.safe_load(m.group(1))
    if isinstance(spec, dict):
        # The notebook contract uses lowercase YAML keys; the runner uppercases
        # them on injection. Models occasionally echo the plan's UPPERCASE
        # variable names back into the YAML keys, so normalize here.
        spec = {k.lower() if isinstance(k, str) else k: v for k, v in spec.items()}
    notes = re.sub(r"```yaml\s*.*?\s*```", "", text, flags=re.DOTALL).strip()
    return ActorResult(spec=spec, notes=notes)
