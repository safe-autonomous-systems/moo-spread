
from problems.zdt_torch import ZDT1, ZDT2, ZDT3
from problems.dtlz_torch import DTLZ2, DTLZ4, DTLZ7
from problems.re_torch import RE21, RE33, RE34, RE37, RE41

def get_problem_torch(name, n_var=30, n_obj=2):
    """
    Get a PyTorch implementation of a multi-objective optimization problem.
    """
    name = name.lower()
    if name == "zdt1":
        return ZDT1(n_var=n_var)
    elif name == "zdt2":
        return ZDT2(n_var=n_var)
    elif name == "zdt3":
        return ZDT3(n_var=n_var)
    elif name == "dtlz2":
        return DTLZ2(n_var=n_var, n_obj=n_obj)
    elif name == "dtlz4":
        return DTLZ4(n_var=n_var, n_obj=n_obj)
    elif name == "dtlz7":
        return DTLZ7(n_var=n_var, n_obj=n_obj)
    elif name == "re21":
        return RE21()
    elif name == "re34":
        return RE34()
    elif name == "re33":
        return RE33()
    elif name == "re37":
        return RE37()
    elif name == "re41":
        return RE41()
    else:
        raise ValueError(f"Unknown problem name: {name}")