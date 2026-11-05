"""Small shared checks for structural Python arguments."""


_MAX_CHUNK_SIZE = (1 << 63) - 1


def _check_chunk_size(value, name='chunk_size', *, allow_none=False):
    if value is None and allow_none:
        return
    if type(value) is not int or value < 1 or value > _MAX_CHUNK_SIZE:
        optional = ' or None' if allow_none else ''
        raise ValueError(
            f'{name} must be a positive integer no greater than {_MAX_CHUNK_SIZE}{optional}')
