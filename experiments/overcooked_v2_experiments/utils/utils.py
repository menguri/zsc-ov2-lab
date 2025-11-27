from functools import partial
import jax


def _check_and_return_outer_dim(args, kwargs, num_mini_batches):
    # 한국어 주석: pmap/vmap 축 길이를 추론할 때, PyTree의 첫 리프가 스칼라인 경우가 있어
    # rank 0 오류가 발생할 수 있습니다. 따라서 rank ≥ 1인 첫 배열 리프를 찾아 그 0번 축을 사용합니다.
    leaves, _ = jax.tree_util.tree_flatten((args, kwargs))
    for leaf in leaves:
        if hasattr(leaf, "shape") and len(leaf.shape) > 0:
            outer_dim = leaf.shape[0]
            break
    else:
        raise ValueError(
            "No array-like argument with rank ≥ 1; cannot determine outer dimension."
        )

    assert (
        outer_dim % num_mini_batches == 0
    ), f"outer_dim {outer_dim} must be divisible by num_mini_batches {num_mini_batches}"
    return outer_dim


def scanned_mini_batch_map(f, num_mini_batches, use_pmap=False, num_devices=None):
    """
    Execute a function in sequential, vmapped mini-batches.
    Enables execution of batches too large to fit in memory.
    """

    map_fn = jax.pmap if use_pmap else jax.vmap
    if num_devices:
        map_fn = partial(mini_batch_pmap, num_mini_batches=num_devices)

    def mapped_fn(*args, **kwargs):
        outer_dim = _check_and_return_outer_dim(args, kwargs, num_mini_batches)
        if outer_dim == num_mini_batches:
            return map_fn(f)(*args, **kwargs)

        def _batched_fn(_, x):
            x_args, x_kwargs = x
            print(f"Args: {x_args}")
            print(f"Kwargs: {x_kwargs}")
            y = map_fn(f)(*x_args, **x_kwargs)
            return None, y

        mini_batched_args, mini_batched_kwargs = jax.tree_util.tree_map(
            lambda x: x.reshape((num_mini_batches, -1, *x.shape[1:])), (args, kwargs)
        )
        _, ret = jax.lax.scan(
            _batched_fn, None, (mini_batched_args, mini_batched_kwargs)
        )
        return jax.tree_util.tree_map(lambda x: x.reshape((outer_dim, *x.shape[2:])), ret)

    return mapped_fn


def mini_batch_pmap(f, num_mini_batches):
    # 한국어 주석: 단일 디바이스(num_mini_batches=1)에서는 pmap 이점이 없고, 축 오류 위험이 커서
    # 그대로 f를 반환해 직접 호출 또는 상위에서 vmap을 사용하도록 합니다.
    if num_mini_batches == 1:
        return f

    def mapped_fn(*args, **kwargs):
        outer_dim = _check_and_return_outer_dim(args, kwargs, num_mini_batches)
        if outer_dim == num_mini_batches:
            # 한국어 주석: outer_dim == num_mini_batches == 1인 경우 pmap 대신 직접 호출하여 rank 0 오류 방지
            if num_mini_batches == 1:
                return f(*args, **kwargs)
            else:
                return jax.pmap(f)(*args, **kwargs)

        # 한국어 주석: (num_mini_batches, -1, ...) 형태로 나눠서 각 디바이스에서 vmap으로 처리
        mini_batched_args, mini_batched_kwargs = jax.tree_util.tree_map(
            lambda x: x.reshape((num_mini_batches, -1, *x.shape[1:])), (args, kwargs)
        )

        ret = jax.pmap(jax.vmap(f))(*mini_batched_args, **mini_batched_kwargs)

        # 한국어 주석: 결과를 다시 (outer_dim, ...)로 결합
        return jax.tree_util.tree_map(lambda x: x.reshape((outer_dim, *x.shape[2:])), ret)

    return mapped_fn
