from scipy.stats import boxcox
from scipy.stats import shapiro

def boxcox_transform(data):

    transformed, lam = boxcox(data)

    return transformed, lam