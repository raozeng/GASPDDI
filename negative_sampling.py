"""Deterministic, unique relation-conditioned negatives for DrugBank/Ryu."""

import random


def _triple_key(a, b, relation):
    # Group reversals for sampling only; preserve direction in returned triples.
    return (a, b, relation) if a < b else (b, a, relation)


def sample_ryu_negatives(smiles1, smiles2, relations, seed=42, max_attempts=256):
    """Return one unique negative per positive, retaining one endpoint and r.

    As in the original sampler, known positives are filtered in both directions.
    This is a conservative sampling policy, not a claim that r is symmetric.
    All randomness is local; sorting the pool removes hash-order dependence.
    Exhausted candidates raise an error rather than permit duplicates or loop.
    """
    if not (len(smiles1) == len(smiles2) == len(relations)):
        raise ValueError("Drug and relation arrays must have equal lengths.")
    if max_attempts < 0:
        raise ValueError("max_attempts must be non-negative.")

    rng = random.Random(seed)
    pool = sorted(set(smiles1) | set(smiles2))
    positives = {
        _triple_key(a, b, r) for a, b, r in zip(smiles1, smiles2, relations)
    }
    used = set()
    neg1, neg2 = [], []

    def allowed(a, b, r):
        key = _triple_key(a, b, r)
        return a != b and key not in positives and key not in used

    for index, (a, b, relation) in enumerate(zip(smiles1, smiles2, relations)):
        candidate = None
        for _ in range(max_attempts):
            replace_second = rng.random() < 0.5
            replacement = rng.choice(pool)
            x, y = (a, replacement) if replace_second else (replacement, b)
            if allowed(x, y, relation):
                candidate = (x, y)
                break

        if candidate is None:
            # Bounded exhaustive fallback checks both allowed corruption sides.
            candidates = pool.copy()
            rng.shuffle(candidates)
            replace_second_first = rng.random() < 0.5
            for replacement in candidates:
                options = [(a, replacement), (replacement, b)]
                if not replace_second_first:
                    options.reverse()
                for x, y in options:
                    if allowed(x, y, relation):
                        candidate = (x, y)
                        break
                if candidate is not None:
                    break

        if candidate is None:
            raise ValueError(
                f"Negative sampling exhausted at positive row {index}, "
                f"relation {relation}: no unused valid one-endpoint corruption. "
                "Cannot maintain 1:1 sampling without relaxing constraints."
            )

        x, y = candidate
        used.add(_triple_key(x, y, relation))
        neg1.append(x)
        neg2.append(y)

    return neg1, neg2
