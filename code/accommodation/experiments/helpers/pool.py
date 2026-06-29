from accommodation.model.accommodation_layer import Pool


def pool(pool_name: str):
    if pool_name == "mean": return Pool.Mean
    elif pool_name == "max": return Pool.Max
    elif pool_name == "mean-max": return Pool.MeanMax
    elif pool_name == "max-mean": return Pool.MaxMean
    elif pool_name == "lse": return Pool.LSE
