from __future__ import division, absolute_import
from six.moves import range

import numpy as np


def do_boxes_intersect(bl, tr):
    (bl1, tr1) = bl
    (bl2, tr2) = tr
    (dimension,) = bl1.shape
    for i in range(0, dimension):
        if max(bl1[i], bl2[i]) > min(tr1[i], tr2[i]):
            return False
    return True


def make_buckets(bottom_left, top_right, allbuckets, max_elements_per_box):
    (dimensions,) = bottom_left.shape

    half = (top_right - bottom_left) / 2.

    def do(dimension, pos):
        if dimension == dimensions:
            origin = bottom_left + pos*half
            bucket = SpatialBinaryTreeBucket(origin, origin + half,
                    max_elements_per_box=max_elements_per_box)
            allbuckets.append(bucket)
            return bucket
        else:
            pos[dimension] = 0
            first = do(dimension + 1, pos)
            pos[dimension] = 1
            second = do(dimension + 1, pos)
            return [first, second]

    return do(0, np.zeros((dimensions,), np.float64))


class SpatialBinaryTreeBucket:
    """This class represents one bucket in a spatial binary tree.
    It automatically decides whether it needs to create more subdivisions
    beneath itself or not.

    .. attribute:: elements

        a list of tuples *(element, bbox)* where bbox is again
        a tuple *(lower_left, upper_right)* of :class:`numpy.ndarray` instances
        satisfying ``(lower_right <= upper_right).all()``.
    """

    def __init__(self, bottom_left, top_right, max_elements_per_box=None):
        """:param bottom_left: A :mod: 'numpy' array of the minimal coordinates
        of the box being partitioned.
        :param top_right: A :mod: 'numpy' array of the maximal coordinates of
        the box being partitioned."""

        self.elements = []

        self.bottom_left = bottom_left
        self.top_right = top_right
        self.center = (bottom_left + top_right) / 2

        # As long as buckets is None, there are no subdivisions
        self.buckets = None
        self.elements = []

        if max_elements_per_box is None:
            dimensions, = self.bottom_left.shape
            max_elements_per_box = 8 * 2**dimensions

        self.max_elements_per_box = max_elements_per_box

    def insert(self, element, bbox):
        """Insert an element into the spatial tree.

        :param element: the element to be stored in the retrieval data
        structure.  It is treated as opaque and no assumptions are made on it.

        :param bbox: A bounding box supplied as a tuple *lower_left,
        upper_right* of :mod:`numpy` vectors, such that *(lower_right <=
        upper_right).all()*.

        Despite these names, the bounding box (and this entire data structure)
        may be of any dimension.
        """

        def insert_into_subdivision(element, bbox):
            bucket_matches = [
                ibucket
                for ibucket, bucket in enumerate(self.all_buckets)
                if do_boxes_intersect((bucket.bottom_left, bucket.top_right), bbox)]

            from random import uniform
            if len(bucket_matches) > len(self.all_buckets) // 2:
                # Would go into more than half of all buckets--keep it here
                self.elements.append((element, bbox))
            elif len(bucket_matches) > 1 and uniform(0, 1) > 0.95:
                # Would go into more than one bucket and therefore may recurse
                # indefinitely. Keep it here with a low probability.
                self.elements.append((element, bbox))
            else:
                for ibucket_match in bucket_matches:
                    self.all_buckets[ibucket_match].insert(element, bbox)

        if self.buckets is None:
            # No subdivisions yet.
            if len(self.elements) > self.max_elements_per_box:
                # Too many elements. Need to subdivide.
                self.all_buckets = []  # noqa: E501 pylint:disable=attribute-defined-outside-init
                self.buckets = make_buckets(
                        self.bottom_left, self.top_right,
                        self.all_buckets,
                        max_elements_per_box=self.max_elements_per_box)

                old_elements = self.elements
                self.elements = []

                # Move all elements from the full bucket into the new finer ones
                for el, el_bbox in old_elements:
                    insert_into_subdivision(el, el_bbox)

                insert_into_subdivision(element, bbox)
            else:
                # Simple:
                self.elements.append((element, bbox))
        else:
            # Go find which sudivision to place element
            insert_into_subdivision(element, bbox)

    def generate_matches(self, point):
        if self.buckets:
            # We have subdivisions. Use them.
            (dimensions,) = point.shape
            bucket = self.buckets
            for dim in range(dimensions):
                if point[dim] < self.center[dim]:
                    bucket = bucket[0]
                else:
                    bucket = bucket[1]

            for result in bucket.generate_matches(point):
                yield result

        # Perform linear search.
        for el, _ in self.elements:
            yield el

    def visualize(self, file):
        file.write("%f %f\n" % (self.bottom_left[0], self.bottom_left[1]))
        file.write("%f %f\n" % (self.top_right[0], self.bottom_left[1]))
        file.write("%f %f\n" % (self.top_right[0], self.top_right[1]))
        file.write("%f %f\n" % (self.bottom_left[0], self.top_right[1]))
        file.write("%f %f\n\n" % (self.bottom_left[0], self.bottom_left[1]))
        if self.buckets:
            for i in self.all_buckets:
                i.visualize(file)

    def plot(self, **kwargs):
        import matplotlib.pyplot as pt
        import matplotlib.patches as mpatches
        from matplotlib.path import Path

        el = self.bottom_left
        eh = self.top_right
        pathdata = [
            (Path.MOVETO, (el[0], el[1])),
            (Path.LINETO, (eh[0], el[1])),
            (Path.LINETO, (eh[0], eh[1])),
            (Path.LINETO, (el[0], eh[1])),
            (Path.CLOSEPOLY, (el[0], el[1])),
            ]

        codes, verts = zip(*pathdata)
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, **kwargs)
        pt.gca().add_patch(patch)

        if self.buckets:
            for i in self.all_buckets:
                i.plot(**kwargs)
