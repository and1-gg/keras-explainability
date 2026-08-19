"""Erweiterte Tests für topological_sort.

topological_sort nimmt eine Adjazenzmatrix (Abhängigkeitsgraph) und gibt
eine topologische Reihenfolge zurück. Diese Reihenfolge stellt sicher, dass
jeder Knoten erst verarbeitet wird, wenn alle seine Abhängigkeiten abgearbeitet
sind – eine Grundvoraussetzung für die korrekte LRP-Rückpropagation.
"""

import numpy as np
import pytest

from explainability.utils import topological_sort


def _is_valid_topological_order(order: np.ndarray,
                                 dependencies: np.ndarray) -> bool:
    """Prüft, ob `order` eine gültige topologische Sortierung von
    `dependencies` ist."""
    position = {node: idx for idx, node in enumerate(order)}
    n = len(dependencies)
    for i in range(n):
        for j in range(n):
            if dependencies[i, j] == 1:
                if position[i] >= position[j]:
                    return False
    return True


def test_topological_sort_linear_chain():
    """Einfache lineare Kette: 0 → 1 → 2 → 3."""
    deps = np.zeros((4, 4), dtype=np.int32)
    deps[0, 1] = 1
    deps[1, 2] = 1
    deps[2, 3] = 1

    order = topological_sort(deps.copy())

    assert list(order) == [0, 1, 2, 3], (
        "topological_sort gibt bei einer linearen Kette falsche Reihenfolge zurück"
    )


def test_topological_sort_single_node():
    """Ein einzelner Knoten ohne Abhängigkeiten."""
    deps = np.zeros((1, 1), dtype=np.int32)
    order = topological_sort(deps.copy())

    assert list(order) == [0]


def test_topological_sort_two_roots():
    """Zwei unabhängige Quellknoten, die in einen gemeinsamen Knoten münden."""
    # 0 → 2, 1 → 2
    deps = np.zeros((3, 3), dtype=np.int32)
    deps[0, 2] = 1
    deps[1, 2] = 1

    order = topological_sort(deps.copy())

    assert len(order) == 3
    assert _is_valid_topological_order(order, deps), (
        "Topologische Reihenfolge ungültig bei zwei Quellknoten"
    )
    # Knoten 2 muss nach 0 und 1 kommen
    pos = {node: idx for idx, node in enumerate(order)}
    assert pos[0] < pos[2]
    assert pos[1] < pos[2]


def test_topological_sort_returns_all_nodes():
    """Die Ausgabe enthält jeden Knoten genau einmal."""
    deps = np.asarray([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    order = topological_sort(deps.copy())

    assert len(order) == 4
    assert set(order) == {0, 1, 2, 3}


def test_topological_sort_diamond():
    """Diamant-Graph: 0 → 1, 0 → 2, 1 → 3, 2 → 3."""
    deps = np.zeros((4, 4), dtype=np.int32)
    deps[0, 1] = 1
    deps[0, 2] = 1
    deps[1, 3] = 1
    deps[2, 3] = 1

    order = topological_sort(deps.copy())

    assert _is_valid_topological_order(order, deps), (
        "Topologische Reihenfolge ungültig bei Diamant-Graph"
    )
    assert order[0] == 0, "Im Diamant-Graph muss Knoten 0 zuerst kommen"
    assert order[-1] == 3, "Im Diamant-Graph muss Knoten 3 zuletzt kommen"
