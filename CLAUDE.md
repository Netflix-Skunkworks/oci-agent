# Observational Causal Inference (OCI) Agent
Parameterizes and executes notebooks, then writes actor-critic reports and suggests next steps.

# Rules
- Do not execute any analysis outside of a notebook.
- Invoke the changing-notebooks skill when changing any notebook. **Always leave an audit trail** (comment-out + inject) for every notebook edit, with no exceptions — even one-line fixes, edits the user requested directly, and edits in areas the user has previously cleaned up. Pruning the audit trail is the user's call, not the agent's.
- Invoke the writing-specs skill when initiating an analysis from a plan or user prompt.
- Invoke the writing-reports skill when reviewing results.
