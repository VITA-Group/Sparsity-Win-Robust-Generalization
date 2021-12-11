import numpy as np
from numpy.core.fromnumeric import shape



def check_valid_sparsities(sparsities):
    if sum([int(s<=1 and s>=0) for s in sparsities])<len(sparsities): print(f"<utils> invalid sparsities {sparsities} encountered.")
    assert sum([int(s<=1 and s>=0) for s in sparsities])==len(sparsities),f"<utils> invalid sparsities {sparsities} encountered."
    return np.array(sparsities)

def bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high):
    lengths_low = [Length/(f_low/area+1)
                   for Length, area in zip(Lengths, areas)]
    overall_sparsity_low = 1-sum(lengths_low)/sum(Lengths)
    if abs(overall_sparsity_low-target_sparsity) < tolerance:
        return [1-length/Length for length, Length in zip(lengths_low, Lengths)]
    lengths_high = [Length/(f_high/area+1)
                    for Length, area in zip(Lengths, areas)]
    overall_sparsity_high = 1-sum(lengths_high)/sum(Lengths)
    if abs(overall_sparsity_high-target_sparsity) < tolerance:
        return [1-length/Length for length, Length in zip(lengths_high, Lengths)]
    force = float(f_low+f_high)/2
    lengths = [Length/(force/area+1) for Length, area in zip(Lengths, areas)]
    overall_sparsity = 1-sum(lengths)/sum(Lengths)
    f_low = force if overall_sparsity < target_sparsity else f_low
    f_high = force if overall_sparsity > target_sparsity else f_high
    return bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high)

def igq_quotas(target_sparsity, shapes, **kwargs):
    counts = [np.prod(shape) for shape in shapes]
    tolerance = 1./sum(counts)
    areas = [1./count for count in counts]
    Lengths = [count for count in counts]
    return bs_force_igq(areas, Lengths, target_sparsity, tolerance, 0, 1e20)

def get_igq_sparsities(model, density):
    sparsity = 1.0 - density
    shapes = [p.shape for p in model.parameters() if len(p.size()) ==4 or len(p.size()) ==2]
    sparsities = check_valid_sparsities(igq_quotas(sparsity, shapes))
    return sparsities