"""Scope guard — LPW is a physics simulator; learning code must not enter.

Mechanical enforcement of the positioning charter (CLAUDE.md): the core
package and examples may consume torch tensors, but must never contain
optimizers, network modules, or backprop. If this test turns red, the
change is out of scope by definition — do not weaken the guard to make it
pass; that decision belongs to the project owner alone.

Runs on CPU, no GPU or engine needed.
"""

import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPES = ("latentphysics", "examples")

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
