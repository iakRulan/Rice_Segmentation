from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: str
    labels: tuple[str, ...]

    @property
    def classes(self) -> int:
        return len(self.labels)


TASKS = {
    "wheat_rape": TaskSpec("wheat_rape", "wheat_rape", ("wheat", "rape")),
    "wheat": TaskSpec("wheat", "wheat_rape", ("wheat",)),
    "rape": TaskSpec("rape", "wheat_rape", ("rape",)),
    "rice": TaskSpec("rice", "rice", ("rice",)),
    # joint: rice + wheat_rape share tile ids (same field, two seasons); predict
    # all three crops from a 6ch dual-temporal image + optional neighbor-row priors.
    "joint": TaskSpec("joint", "rice", ("wheat", "rape", "rice")),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}") from exc
