from scipy.stats import shapiro


def shapiro_test(data):

    stat, p = shapiro(data)

    return stat, p