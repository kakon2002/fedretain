Starter kit for the fedretain repository.

    legacy/fedfreq_fix.py      version 2026-10-20g   (training, evaluation, statistics; produced the paper's numbers)
    legacy/fedfreq_extra.py    version 2026-10-20x4  (subset ablations, FedPer helpers, figures)
    legacy/tests/              three smoke tests on synthetic data, all passing; see legacy/tests/README.md
    CLAUDE_CODE_PROMPT.md      the instructions for Claude Code to build the clean package from these

Verified before packaging: all three tests pass from a neutral working directory, and the training loop is
bit-for-bit deterministic on CPU (same seed twice gives identical weights and histories), so the equivalence
test demanded in CLAUDE_CODE_PROMPT.md is achievable.
