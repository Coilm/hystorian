def gauss_area(x, y):
    """
    Determine the area created by the polygon formed by x,y using the Gauss's area formula (also
    called shoelace formula)

    Parameters
    ----------
    x  : array_like
        values along the x axis
    y : array_like
        values along the x axis

    Returns
    -------
        float
            value corresponding at the encompassed area
    """

    area = 0.0
    for i in range(len(x)):
        x1 = x[i]
        y1 = y[i]

        if i < len(x) - 1:
            x2 = x[i + 1]
            y2 = y[i + 1]
        else:
            x2 = x[0]
            y2 = y[0]

        area = area + x1 * y2 - x2 * y1
    return abs(area / 2.0)



