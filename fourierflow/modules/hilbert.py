import numpy as np


class HilbertCurve:
    def __init__(self, nw=None, ne=None, sw=None, se=None, shape=None, leaf=None):
        self.nw = nw
        self.ne = ne
        self.sw = sw
        self.se = se
        self.shape = shape
        self.direction = 1
        self.leaf = leaf
        if shape:
            self.change_shape(shape, direction=1)

    def change_shape(self, shape, direction):
        self.shape = shape
        self.direction = direction
        if shape == 'D':
            if self.nw:
                self.nw.change_shape('U', 1)
            if self.ne:
                self.ne.change_shape('D', 1)
            if self.sw:
                self.sw.change_shape('N', -1)
            if self.se:
                self.se.change_shape('D', 1)

        elif shape == 'U':
            if self.nw:
                self.nw.change_shape('D', 1)
            if self.ne:
                self.ne.change_shape('E', -1)
            if self.sw:
                self.sw.change_shape('U', 1)
            if self.se:
                self.se.change_shape('U', 1)

        elif shape == 'N':
            if self.nw:
                self.nw.change_shape('N', 1)
            if self.ne:
                self.ne.change_shape('N', 1)
            if self.sw:
                self.sw.change_shape('D', -1)
            if self.se:
                self.se.change_shape('E', 1)

        elif shape == 'E':
            if self.nw:
                self.nw.change_shape('E', 1)
            if self.ne:
                self.ne.change_shape('U', -1)
            if self.sw:
                self.sw.change_shape('E', 1)
            if self.se:
                self.se.change_shape('N', 1)

        else:
            raise ValueError

    def get_path(self):
        if self.leaf is not None:
            return self.leaf
        nw = self.nw.get_path() if self.nw else []
        ne = self.ne.get_path() if self.ne else []
        sw = self.sw.get_path() if self.sw else []
        se = self.se.get_path() if self.se else []

        if self.shape == 'D':
            path = nw + ne + se + sw
        elif self.shape == 'U':
            path = nw + sw + se + ne
        elif self.shape == 'N':
            path = sw + nw + ne + se
        elif self.shape == 'E':
            path = ne + nw + sw + se
        else:
            raise ValueError

        if self.direction == -1:
            path = list(reversed(path))

        return path


def linearize(indices, mesh_pos, shape):
    if len(indices) == 1:
        return HilbertCurve(leaf=indices)
    elif len(indices) == 0:
        return None

    positions = np.array(mesh_pos[indices])

    x_min = positions[:, 0].min()
    x_max = positions[:, 0].max()
    x_mid = (x_min + x_max) / 2

    y_min = positions[:, 1].min()
    y_max = positions[:, 1].max()
    y_mid = (y_min + y_max) / 2

    # Divide into four quadrants
    north_west, north_east, south_west, south_east = [], [], [], []
    for i, coords in zip(indices, positions):
        if coords[0] < x_mid and coords[1] < y_mid:
            south_west.append(i)
        elif coords[0] < x_mid and coords[1] >= y_mid:
            north_west.append(i)
        elif coords[0] >= x_mid and coords[1] < y_mid:
            south_east.append(i)
        elif coords[0] >= x_mid and coords[1] >= y_mid:
            north_east.append(i)
        else:
            raise ValueError

    # For each quadrant, find the optimal curve
    nw = linearize(north_west, mesh_pos, shape)
    ne = linearize(north_east, mesh_pos, shape)
    sw = linearize(south_west, mesh_pos, shape)
    se = linearize(south_east, mesh_pos, shape)
    curve = HilbertCurve(nw, ne, sw, se, shape)

    return curve
