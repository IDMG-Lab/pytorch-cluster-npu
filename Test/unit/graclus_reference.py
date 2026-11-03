"""Independent CPU specification used by the repository's Graclus tests.

The server test file imports this helper, but the supplied checkout omitted it.
This helper is used only for correctness, outside every performance measurement.
"""
import torch


def phase_reference(rowptr, col, weight, state):
    rp, ci = rowptr.tolist(), col.tolist()
    w = None if weight is None else weight.tolist()
    candidate = [-1] * len(state)
    for u, label in enumerate(state):
        if label >= 0:
            continue
        winner = None
        for edge in range(rp[u], rp[u + 1]):
            v = ci[edge]
            if v == u or state[v] >= 0 or (w is not None and not w[edge] >= 0):
                continue
            key = ((0 if w is None else w[edge]), max(u, v), min(u, v))
            if winner is None or key > winner:
                candidate[u], winner = v, key
    after = list(state)
    for u, v in enumerate(candidate):
        if state[u] < 0:
            if v < 0:
                after[u] = u
            elif candidate[v] == u:
                after[u] = min(u, v)
    return candidate, after


def matching_reference(rowptr, col, weight=None):
    n = rowptr.numel() - 1
    state = [-1] * n
    for _ in range(64):
        _, after = phase_reference(rowptr, col, weight, state)
        if after == state or all(v >= 0 for v in after):
            state = after
            break
        state = after
    rp, ci = rowptr.tolist(), col.tolist()
    w = None if weight is None else weight.tolist()
    for u in range(n):
        if state[u] >= 0:
            continue
        options = []
        for edge in range(rp[u], rp[u + 1]):
            v = ci[edge]
            if v == u or state[v] >= 0 or (w is not None and not w[edge] >= 0):
                continue
            reverse = any(ci[j] == u and (w is None or w[j] >= 0)
                          for j in range(rp[v], rp[v + 1]))
            if reverse:
                options.append(((0 if w is None else w[edge]), max(u, v), min(u, v), v))
        v = max(options)[-1] if options else u
        state[u] = state[v] = min(u, v)
    return torch.tensor(state, dtype=torch.int64)


def assert_matching(rowptr, col, weight, output, maximal=True):
    output = output.cpu()
    n = rowptr.numel() - 1
    assert output.dtype == torch.int64 and output.shape == (n,)
    assert bool(((output >= 0) & (output < n)).all())
    counts = torch.bincount(output, minlength=n)
    assert int(counts.max()) <= 2 if n else True
    paired = (counts[output] == 2).nonzero().flatten()
    rp, ci = rowptr.tolist(), col.tolist()
    w = None if weight is None else weight.tolist()
    groups = {}
    for u in paired.tolist():
        groups.setdefault(int(output[u]), []).append(u)
    for label, (u, v) in groups.items():
        assert label == min(u, v)
        for a, b in ((u, v), (v, u)):
            assert any(ci[j] == b and (w is None or w[j] >= 0)
                       for j in range(rp[a], rp[a + 1]))
    if maximal:
        singles = (counts[output] == 1).tolist()
        for u in range(n):
            if not singles[u]:
                continue
            assert int(output[u]) == u
            for j in range(rp[u], rp[u + 1]):
                v = ci[j]
                if v != u and singles[v] and (w is None or w[j] >= 0):
                    assert not any(ci[k] == u and (w is None or w[k] >= 0)
                                   for k in range(rp[v], rp[v + 1]))
