from numpy.random import default_rng

def generateB(rng):
    # 使用传入的 rng
    rng = default_rng(42)
    return rng.normal(size=3)
