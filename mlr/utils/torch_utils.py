from functools import reduce


def get_total_params(m):
    return reduce(
        lambda x, y: x + y,
        [reduce(lambda x, y: x * y, p.shape) for p in m.parameters()],
    )
