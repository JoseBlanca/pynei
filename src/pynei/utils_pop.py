from typing import Sequence

from pynei.config import DEF_POP_NAME

# The pops are always given as a dict of pop name -> sample names, so that the
# pops can have a name. The translation into the sample idxs that the
# calculations need is done here, and only here.
Pops = dict[str, Sequence[str]]


def _calc_pops_idxs(pops: Pops | None, samples):
    if pops is None:
        return {DEF_POP_NAME: slice(None, None)}

    sample_idxs = {sample: idx for idx, sample in enumerate(samples)}

    pops_idxs = {}
    for pop_id, pop_samples in pops.items():
        if isinstance(pop_samples, (str, bytes)) or not isinstance(
            pop_samples, Sequence
        ):
            raise ValueError(
                f"The samples of the pop {pop_id} should be a sequence of sample names, but they are: {pop_samples!r}"
            )
        missing_samples = [
            sample for sample in pop_samples if sample not in sample_idxs
        ]
        if missing_samples:
            raise ValueError(
                f"These samples of the pop {pop_id} are not in the variants: {missing_samples}"
            )
        pops_idxs[pop_id] = [sample_idxs[sample] for sample in pop_samples]
    return pops_idxs
