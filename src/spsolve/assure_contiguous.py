from numpy import ndarray

from .spparms import _PREFER_ORDER


def assure_contiguous(a: ndarray) -> ndarray:
    return a if a.data.contiguous else a.copy(order=_PREFER_ORDER)


if __name__ == "__main__":
    pass