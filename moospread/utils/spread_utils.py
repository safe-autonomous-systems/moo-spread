from torch.utils.data import DataLoader, TensorDataset
import dis
import inspect
import ast
import textwrap

def get_ddpm_dataloader(X,
                        y,
                        validation_split=0.1, 
                        batch_size=32):
    val_dataloader = None
    if validation_split > 0.0:
        train_size = int(X.shape[0] - int(X.shape[0] * validation_split))
        X_val = X[train_size:]
        y_val = y[train_size:]
        X = X[:train_size]
        y = y[:train_size]

        tensor_x_val = X_val.float() 
        tensor_y_val = y_val.float() 
        dataset_val = TensorDataset(tensor_x_val, tensor_y_val)
        val_dataloader = DataLoader(
                    dataset_val,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )
    tensor_x = X.float() 
    tensor_y = y.float() 
    dataset_train = TensorDataset(tensor_x, tensor_y)
    train_dataloader = DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
    
    return train_dataloader, val_dataloader

def is_pass_function(func) -> bool:
    """
    Return True iff the function body is effectively just `pass`
    (optionally with a docstring). Works on Python 3.8–3.12+.
    """
    # -------- Try AST first (most reliable) --------
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        src = None

    if src:
        mod = ast.parse(textwrap.dedent(src))
        fn = next((n for n in ast.walk(mod)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if fn:
            body = list(fn.body)
            # drop docstring if present
            if body and isinstance(body[0], ast.Expr) and \
               isinstance(getattr(body[0], "value", None), ast.Constant) and \
               isinstance(body[0].value.value, str):
                body = body[1:]
            # True if remaining stmts are all Pass (or empty)
            return all(isinstance(n, ast.Pass) for n in body)

    # -------- Fallback: bytecode pattern (version-tolerant) --------
    instrs = list(dis.get_instructions(func))

    # remove version-specific noise
    noise = {"RESUME", "CACHE", "EXTENDED_ARG", "NOP"}
    core = [i for i in instrs if i.opname not in noise]

    # strip docstring store: LOAD_CONST <str>; STORE_* __doc__
    i = 0
    while i + 1 < len(core):
        a, b = core[i], core[i + 1]
        if (a.opname == "LOAD_CONST" and isinstance(a.argval, str)
            and b.opname in {"STORE_NAME", "STORE_FAST"} and b.argval == "__doc__"):
            del core[i:i+2]
            continue
        i += 1

    # Accept either:
    #  - LOAD_CONST None; RETURN_VALUE
    #  - RETURN_CONST None   (3.12+)
    if len(core) == 2 and core[0].opname == "LOAD_CONST" and core[0].argval is None \
       and core[1].opname == "RETURN_VALUE":
        return True
    if len(core) == 1 and core[0].opname == "RETURN_CONST" and core[0].argval is None:
        return True

    return False