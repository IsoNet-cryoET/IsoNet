from __future__ import absolute_import
import numpy as np
from six.moves import range
from six.moves import zip


# {{{ eoc estimation --------------------------------------------------------------

def estimate_order_of_convergence(abscissae, errors):
    """Assuming that abscissae and errors are connected by a law of the form

    error = constant * abscissa ^ (order),

    this function finds, in a least-squares sense, the best approximation of
    constant and order for the given data set. It returns a tuple (constant, order).
    """
    assert len(abscissae) == len(errors)
    if len(abscissae) <= 1:
        raise RuntimeError("Need more than one value to guess order of convergence.")

    coefficients = np.polyfit(np.log10(abscissae), np.log10(errors), 1)
    return 10**coefficients[-1], coefficients[-2]


class EOCRecorder(object):
    def __init__(self):
        self.history = []

    def add_data_point(self, abscissa, error):
        self.history.append((abscissa, error))

    def estimate_order_of_convergence(self, gliding_mean=None):
        abscissae = np.array([a for a, e in self.history])
        errors = np.array([e for a, e in self.history])

        size = len(abscissae)
        if gliding_mean is None:
            gliding_mean = size

        data_points = size - gliding_mean + 1
        result = np.zeros((data_points, 2), float)
        for i in range(data_points):
            result[i, 0], result[i, 1] = estimate_order_of_convergence(
                abscissae[i:i+gliding_mean], errors[i:i+gliding_mean])
        return result

    def order_estimate(self):
        return self.estimate_order_of_convergence()[0, 1]

    def max_error(self):
        return max(err for absc, err in self.history)

    def pretty_print(self,
            abscissa_label="h",
            error_label="Error",
            gliding_mean=2,
            abscissa_format="%s",
            error_format="%s",
            eoc_format="%s"):
        from pytools import Table

        tbl = Table()
        tbl.add_row((abscissa_label, error_label, "Running EOC"))

        gm_eoc = self.estimate_order_of_convergence(gliding_mean)
        for i, (absc, err) in enumerate(self.history):
            absc_str = abscissa_format % absc
            err_str = error_format % err
            if i < gliding_mean-1:
                eoc_str = ""
            else:
                eoc_str = eoc_format % (gm_eoc[i - gliding_mean + 1, 1])

            tbl.add_row((absc_str, err_str, eoc_str))

        if len(self.history) > 1:
            return "%s\n\nOverall EOC: %s" % (str(tbl),
                    self.estimate_order_of_convergence()[0, 1])
        else:
            return str(tbl)

    def __str__(self):
        return self.pretty_print()

    def write_gnuplot_file(self, filename):
        outfile = open(filename, "w")
        for absc, err in self.history:
            outfile.write("%f %f\n" % (absc, err))
        result = self.estimate_order_of_convergence()
        const = result[0, 0]
        order = result[0, 1]
        outfile.write("\n")
        for absc, err in self.history:
            outfile.write("%f %f\n" % (absc, const * absc**(-order)))

# }}}


# {{{ p convergence verifier

class PConvergenceVerifier(object):
    def __init__(self):
        self.orders = []
        self.errors = []

    def add_data_point(self, order, error):
        self.orders.append(order)
        self.errors.append(error)

    def __str__(self):
        from pytools import Table
        tbl = Table()
        tbl.add_row(("p", "error"))

        for p, err in zip(self.orders, self.errors):
            tbl.add_row((str(p), str(err)))

        return str(tbl)

    def __call__(self):
        orders = np.array(self.orders, np.float64)
        errors = np.abs(np.array(self.errors, np.float64))

        rel_change = np.diff(1e-20 + np.log10(errors)) / np.diff(orders)

        assert (rel_change < -0.2).all()

# }}}


# vim: foldmethod=marker
