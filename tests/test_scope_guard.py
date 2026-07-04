"""Scope guard — LPW is a physics simulator; learning code must not enter.

Mechanical enforcement of the positioning charter (CLAUDE.md): the core
package and examples may consume torch tensors, but must never contain
optimizers, network modules, or backprop — except under
`latentphysics/neural/`, the single owner-signed carve-out for learned
simulation of whitelisted physics (deformables/fluids/aerodynamics; see
CLAUDE.md). Behavior-code signatures stay banned even there. If this test
turns red, the change is out of scope by definition — do not weaken the
guard to make it pass; that decision belongs to the project owner alone.

Runs on CPU, no GPU or engine needed.
"""

import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPES = ("latentphysics", "examples")

# Owner-signed carve-out (standalone change, 2026-07-03): latentphysics/neural
# is the ONLY sanctioned home for learned simulation of whitelisted physics
# (deformables / fluids / aerodynamics). Exactly one entry, ever; adding a
# second or widening this one is a standalone owner-approved change.
LEARNING_ALLOWED_DIRS = (os.path.join(ROOT, "latentphysics", "neural"),)


def _in_allowed_dir(path):
    p = os.path.abspath(path)
    return any(p.startswith(d + os.sep) for d in LEARNING_ALLOWED_DIRS)

# training-code signatures; benign tensor math never needs any of these
FORBIDDEN_PATTERNS = (
    "torch.optim",
    "torch.nn",
    "from torch import nn",
    ".backward(",
    "requires_grad",
    "Adam(",
    "SGD(",
)
FORBIDDEN_MODULE_NAMES = ("rl", "policy", "policies", "agents", "train", "learn")


def _py_files():
    for scope in SCOPES:
        yield from glob.glob(os.path.join(ROOT, scope, "**", "*.py"), recursive=True)


def test_no_learning_code():
    offenders = []
    for path in _py_files():
        if _in_allowed_dir(path):
            continue  # the single owner-signed carve-out (see header comment)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        for pat in FORBIDDEN_PATTERNS:
            if pat in src:
                offenders.append(f"{os.path.relpath(path, ROOT)}: {pat!r}")
    assert not offenders, (
        "training-code signatures found — LPW is a simulator, training is "
        "the user's side of the API (see CLAUDE.md):\n" + "\n".join(offenders)
    )


def test_no_learning_modules():
    pkg = os.path.join(ROOT, "latentphysics")
    bad = [e for e in os.listdir(pkg)
           if os.path.splitext(e)[0].lower() in FORBIDDEN_MODULE_NAMES]
    assert not bad, f"forbidden module names in core package: {bad}"


def test_charter_exists():
    assert os.path.exists(os.path.join(ROOT, "CLAUDE.md")), (
        "scope charter (CLAUDE.md) is missing — it is what keeps every "
        "session aligned to the simulator-only positioning"
    )


def test_carveout_is_exactly_neural():
    """The allowlist is one hard-coded path; widening it must be a visible,
    owner-approved diff to BOTH this tuple and this test."""
    assert LEARNING_ALLOWED_DIRS == (os.path.join(ROOT, "latentphysics", "neural"),)


def test_neural_no_behavior_code():
    """Even inside the carve-out, behavior learning stays banned: learned
    SIMULATION of whitelisted physics is sanctioned, policies are not."""
    banned_names = ("rl", "policy", "policies", "agents")
    banned_signatures = ("PPO", "reward_shap", "replay_buffer")
    offenders = []
    for dirpath, dirnames, filenames in os.walk(LEARNING_ALLOWED_DIRS[0]):
        for entry in dirnames + filenames:
            if os.path.splitext(entry)[0].lower() in banned_names:
                offenders.append(f"name: {entry}")
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                src = f.read()
            offenders += [f"{fn}: {p!r}" for p in banned_signatures if p in src]
    assert not offenders, "behavior code inside neural/ carve-out:\n" + "\n".join(offenders)


def test_neural_charter_docstring():
    """The constitutional docstring is load-bearing: neural/ must state its
    own limits in the exact terms the charter uses."""
    init = os.path.join(LEARNING_ALLOWED_DIRS[0], "__init__.py")
    with open(init, encoding="utf-8") as f:
        src = f.read()
    for phrase in ("simulation method", "never behavior", "whitelisted", "classical"):
        assert phrase in src, f"neural/__init__.py docstring lost its charter phrase {phrase!r}"
