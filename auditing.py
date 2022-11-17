"""Tools for the auditing of synthetic data."""

import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools
import scipy
import scipy.optimize


### Function definitions and helpers ###


class Function:

    # The main thing a function does is to map a (N,d) matrix to a vector (N,).
    def __call__(self, x):
        abstract

    # Also, we add support for basic algebraic manipulations
    def __add__(self, function):
        return AddedFunctions(self, function)

    def __sub__(self, function):
        return AddedFunctions(self, MultipliedFunction(function, -1))

    def __mul__(self, factor):
        return MultipliedFunction(self, factor)

    def __rmul__(self, factor):
        return self.__mul__(factor)

    # Support for nice displays.
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "F"


class AddedFunctions(Function):
    """Helper for Function: f1 + ... + fk."""

    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, x):
        y = [f(x) for f in self.functions]
        return np.sum(y, axis=0)

    def __str__(self):
        return "(" + ("+".join([str(f) for f in self.functions])) + ")"


class MultipliedFunction(Function):
    """Helper for Function: a * f (where a is constant)."""

    def __init__(self, function, a):
        self.function = function
        self.a = a

    def __call__(self, x):
        y = self.function(x)
        return self.a * y

    def __str__(self):
        return f"{self.a:.2e}*{str(self.function)}"


def split_linear_combination(large_function, tol=1e-16):
    """Decompose a complex function in a linear combination of base factors."""
    # For each basis element (function), compute a factor (starts at 0).
    factors = collections.defaultdict(lambda: 0)
    L = [(large_function, 1)]
    while L:
        f, scale = L.pop()
        if isinstance(f, AddedFunctions):
            for g in f.functions:
                L.append((g, scale))
        elif isinstance(f, MultipliedFunction):
            L.append((f.function, f.a * scale))
        else:
            factors[f] = factors[f] + scale
    # Filter out the zeroes.
    zero_factors = []
    for f, a in factors.items():
        if np.abs(a) < tol:
            zero_factors.append(f)
    for f in zero_factors:
        del factors[f]
    return factors


def refactor_linear_combination(large_function, tol=1e-16):
    """Simplify an expression combining AddedFunction and MultipliedFunction."""
    # For each basis element (function), compute a factor (starts at 0).
    factors = split_linear_combination(large_function, tol)
    # This is possibly zero! In this case, return 0.
    if len(factors) == 0:
        return ConstantFunction(0)
    # Regroup the factors togethers.
    multiply = lambda f, a: MultipliedFunction(f, a) if np.abs(a - 1) > 1e-10 else f
    if len(factors) > 1:
        return AddedFunctions(*[multiply(f, a) for f, a in factors.items()])
    else:
        f, a = list(factors.items())[0]
        return multiply(f, a)


### Some useful functions ###


class ConstantFunction(Function):
    """Function with constant value x -> c."""

    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        x = np.array(x)
        return np.full(x.shape[0], self.value)

    def __str__(self):
        return str(self.value)


class Delta(Function):
    """Dirac's delta: x -> I{x = t} for a given t."""

    def __init__(self, position):
        self.position = np.array(position)  # A vector of length k.

    def __call__(self, x):
        return np.all(np.array(x) == self.position, axis=1) * 1.0

    def __str__(self):
        return f"delta({self.position})"


class MarginalBin(Function):
    """Element of a histogram: single bin on attribute i."""

    def __init__(self, attributes, values):
        """Both attributes and values are lists of up to k elements. This returns 1 for 
           each x such that (x[attributes[i]] in values_i) for all i."""
        self.attributes = attributes
        # Correct values: if singular value, make it a list.
        self.values = [[v] if isinstance(v, int) else v for v in values]

    def __call__(self, x):
        x = np.array(x)
        answers = [np.isin(x[:, a], v) for a, v in zip(self.attributes, self.values)]
        return np.prod(answers, axis=0)

    def __str__(self):
        return "{%s}" % (
            "^".join([f"x_{a}\\in{v}" for a, v in zip(self.attributes, self.values)])
        )


### Scalar Product: definition and implementation ###


class ScalarProduct:
    def __call__(self, f, g):
        """Takes two functions as arguments and returns a real."""
        abstract


class NaiveScalarProduct(ScalarProduct):
    """The basic implementation of the scalar product is by iterating over all
       possible records. This is correct, but tremendously slow."""

    def __init__(self, domain_sizes):  # List of num_values per attribute.
        self.k = len(domain_sizes)
        self.domain_sizes = domain_sizes
        self.scale = 1 / np.prod(self.domain_sizes)

    def __call__(self, f, g):
        result = 0
        for v in itertools.product(*[range(ni) for ni in self.domain_sizes]):
            result += f([v]) * g([v])
        return result[0] * self.scale


class RestrictedScalarProduct(ScalarProduct):
    """
    In practice, for auditing, we only want to compute scalar products over
    the support of the original distribution. This support is expressed as a
    dataset of size $|supp(p_d)| * k$ of integers.

    Importantly, this is only a scalar product over distributions taking
    nonzero values for values in the records.
    """

    def __init__(self, records):
        self.records = np.array(records)
        # TODO: check uniqueness

    def __call__(self, f, g):
        return np.mean(
            f(self.records) * g(self.records)
        )  # TODO: scale ? Use total universe size ?


class DeltaScalarProduct(ScalarProduct):
    """Scalar product where one of the functions is a delta."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, f, g):
        if isinstance(f, Delta):
            return g(f.position.reshape(1, -1))[0] * self.scale
        elif isinstance(g, Delta):
            return f(g.position.reshape(1, -1))[0] * self.scale
        else:
            raise Exception("Incompatible functions, deltas required!")


class HistogramScalarProduct(ScalarProduct):
    """Scalar product where both functions are histogram bins."""

    def __init__(self, domain_sizes):
        self.domain_sizes = domain_sizes

    def __call__(self, f, g):
        if not isinstance(f, MarginalBin) or not isinstance(f, MarginalBin):
            raise Exception("This requires two MarginalBin functions.")
        # First, get the list of columns concerned and corresponding values.
        attr_value_map = dict(zip(f.attributes, f.values))
        overlap = 1
        for a, v in zip(g.attributes, g.values):
            # If the histograms concern the same attribute, check how many values
            # are overlapping between the two. This scales the total scalar product.
            if a in attr_value_map:
                # Compute the number of overlapping values in the bin.
                overlap_a = len(set(v).intersection(attr_value_map[a]))
                if not overlap_a:
                    return 0
                # Scale the volume (overlap) covered by this bin.
                overlap = overlap * overlap_a
            attr_value_map[a] = v
        # Then, compute the scalar product, scaled by 1 / |X|. Since the product assumes
        # uniformity of the space, this is prod_{i not in f U g} n_i / |X| = 1 / prod_{i in f U g} n_i.
        return overlap / np.prod([self.domain_sizes[a] for a in attr_value_map.keys()])

class SmartScalarProduct(ScalarProduct):
    """This scalar product groups together function-specific products."""

    def __init__(self, domain_sizes, restrict_to_records=None):
        """If restrict_to_records != None, use these records for the scalar product."""
        self.domain_sizes = domain_sizes
        self.restrict_to_records = restrict_to_records
        # Set the "default" scalar product.
        if restrict_to_records is not None:
            self._default_product = RestrictedScalarProduct(restrict_to_records)
        else:
            self._default_product = NaiveScalarProduct(domain_sizes)
        # Initialise smarter scalar products.
        self.delta_product = DeltaScalarProduct(1 / np.prod(domain_sizes))
        self.marginal_product = HistogramScalarProduct(domain_sizes)

    def __call__(self, f, g):
        print(type(f), type(g))
        print(isinstance(f, MarginalBin), isinstance(g, MarginalBin))
        if isinstance(f, Delta) or isinstance(g, Delta):
            return self.delta_product(f, g)
        if isinstance(f, MarginalBin) and isinstance(g, MarginalBin):
            return self.marginal_product(f, g)
        return self._default_product(f, g)


class LinearCompositionScalarProduct(ScalarProduct):
    """This scalar product extends linear composition of scalar products."""

    def __init__(self, base_scalar_product):
        self.base_scalar_product = base_scalar_product

    def __call__(self, f, g):
        if isinstance(f, AddedFunctions):
            return sum([self(f_i, g) for f_i in f.functions])
        elif isinstance(f, MultipliedFunction):
            return f.a * self(f.function, g)
        elif isinstance(g, AddedFunctions):
            return sum([self(f, g_i) for g_i in g.functions])
        elif isinstance(g, MultipliedFunction):
            return g.a * self(f, g.function)
        else:
            return self.base_scalar_product(f, g)


### Gram-Schmidt procedure and nice tools.


def gram_schmidt(functions, scalar_product, tol=1e-10, iterator=lambda x: x):
    basis = []
    for f in iterator(functions):
        # Orthogonalise the element from the existing basis.
        # Due to the way scalar products are implemented (especially linear
        # operations), this is optimised by first computing all scalar products.
        scalar_product_basis = [scalar_product(f, b) for b in basis]
        # We want to compute the orthogonalised vector (from basis b):
        #    f_tilde = f - sum_i <b_i, f> * b_i
        # The norm is ||f_tilde||^2 = <f, f> - sum_i <b_i, f>^2.
        f_tilde = f
        for s, b in zip(scalar_product_basis, basis):
            if np.abs(s) > tol:
                f_tilde = f_tilde - s * b
        # Compute the norm, using the fact that the basis is orthonormal.
        f_tilde_squared_norm = (
            scalar_product(f, f) - (np.array(scalar_product_basis) ** 2).sum()
        )
        # Check that the vector is not 0 (with some tolerance).
        if f_tilde_squared_norm >= tol:
            f_tilde_norm = np.sqrt(f_tilde_squared_norm)
            f_tilde = f_tilde * (1 / f_tilde_norm)
            # Simplify the (potentially complex) function.
            f_tilde = refactor_linear_combination(f_tilde)
            basis.append(f_tilde)
    return basis


## Gram-Schmidt is very slow in practice, due to the way scalar products are
#  computed. We here propose an optimised orthogonal space.


class OrthogonalSpace:
    def __init__(self, scalar_product, generating_functions, tol=1e-10):
        self.scalar_product = scalar_product
        self.generating_functions = generating_functions
        self.k = len(self.generating_functions)
        self.tol = tol
        # Construct the basis using internal methods.
        self._compute_scalar_product_matrix()
        self._gram_schmidt()
        self._reconstruct_basis()

    # Internal methods.
    def _compute_scalar_product_matrix(self):
        # First, compute all pairwise scalar products.
        self.A = np.zeros((self.k, self.k))
        for i, f_i in enumerate(self.generating_functions):
            for j, f_j in enumerate(self.generating_functions[i:]):
                self.A[i, i + j] = self.A[i + j, i] = self.scalar_product(f_i, f_j)

    def _gram_schmidt(self):
        # Then, compute the basis in terms of combinations of theta.
        self.theta = np.zeros((self.k, self.k))
        for l in range(self.k):
            sigma = np.zeros((l,))
            for i in range(l):  # Up to k-1.
                sigma[i] = np.sum(self.A[l, : i + 1] * self.theta[i, : i + 1])
            norm = self.A[l, l] - np.sum(sigma ** 2)
            if norm >= self.tol:
                self.theta[l, l] = 1
                for j in range(l):
                    self.theta[l, j] = -np.sum(self.theta[j:l, j] * sigma[j:l])
                self.theta[l, :] = self.theta[l, :] / np.sqrt(norm)

    def _reconstruct_basis(self):
        # Finally, re-construct the basis (for fun).
        self.basis = []
        for l in range(self.k):
            if np.abs(self.theta[l, l]) < self.tol:
                continue
            f = self.theta[l, l] * self.generating_functions[l]
            for i in range(l):
                if np.abs(self.theta[l, i]) > self.tol:
                    f = f + self.theta[l, i] * self.generating_functions[i]
            self.basis.append(f)

    # Public methods to use this space.
    def project(self, g):
        """Project a function over the space."""
        # First, compute the scalar product between g and all base functions f_j.
        scalar_product_g_fi = np.zeros((self.k,))
        for i in range(self.k):
            scalar_product_g_fi[i] = self.scalar_product(
                self.generating_functions[i], g
            )
        # Then, compute the scaling factor <g, b_i>.
        sigma = np.zeros((self.k,))
        for i in range(self.k):
            sigma[i] = np.sum(self.theta[i, : i + 1] * scalar_product_g_fi[: i + 1])
        # Finally, compute - <g, b_i> * b_i.
        functions_to_substract = []
        for j in range(self.k):
            scaling = np.sum(self.theta[j:, j] * sigma[j:])
            if np.abs(scaling) > self.tol:
                functions_to_substract.append(self.generating_functions[j] * scaling)
        # Group together the scaled base functions.
        return AddedFunctions(*functions_to_substract)

    def orthogonalise(self, g):
        """Remove Proj(g) from g (orthogonalise g from the space)."""
        g_proj = self.project(g)
        return g - g_proj

    # This can also serve as a scalar product, which is nice.
    def __call__(self, f, g):
        return self.scalar_product(f, g)


## Tools for the auditing procedure on a reduced dataset.


class Auditor:
    """
    Auditor to generate dataset with different values of a target function g
    while maintaining all other entries constant.

    """

    def __init__(self, phi, dataset, subsample_size=1000, use_lp=True):
        """
        Parameters
        ----------
        phi: list of Function
            The functions to maintain constant across datasets.
        dataset: a pandas DataFrame or np.array.
            A list of feasible points from which to subsample.
        subsample_size: a positive integer.
            The size of the subspace to use for datasets and scalar product.
        use_lp: boolean (default True).
            Whether to use linear programming to find extremal datasets.

        """
        # Save the parameters.
        self.phi = phi
        self.dataset = dataset
        self.subsample_size = subsample_size
        self.use_lp = use_lp
        # Create a subset of the unique records in the datasets.
        self.records = (
            dataset.drop_duplicates(inplace=False)
            .sample(subsample_size, replace=False)
            .values
        )
        # Initialise a scalar product on this subset.
        self.scalar_product = RestrictedScalarProduct(self.records)
        # Finally, initialise an orthogonal basis over this set.
        # Note that we cannot reuse another orthogonal space, since this
        # one uses the "local" scalar product for extremal datasets.
        self.space = OrthogonalSpace(
            self.scalar_product,
            # We add the constant function 1 in case it is not generated
            # by phi.
            self.phi + [ConstantFunction(1)],
        )

    def generate_extremal_datasets(self, g, num_repeats=100, scaling=1):
        """
        Parameters
        ----------
        g: Function
            The function for which extremal datasets must be generated.
        num_repeats: int
            Number of repetitions of each record in the starting dataset. The
            size of output datasets is len(self.records) * num_repeats. Hence,
            higher num_repeats gives better accuracy but is more expensive.
        scaling: float in (0,1]
            The scaling of the move. If scaling=1, then the datasets produced
            are extremal (as min/max g as possible). If scaling is close to 0,
            then the datasets are very close to one another. This only applies
            if use_lp = False.

        Returns
        -------
        Either a pair (d-, d+), where g(d-) <=  g(d+), where g(d-) (resp. g(d+))
            is close to the minimum (resp. maximum);
        or (None, None), if g is determined uniquely by phi.

        """
        # First, orthogonalise the function g with respect to phi.
        g_perp = self.space.orthogonalise(g)
        norm_squared = self.scalar_product(g_perp, g_perp)
        if norm_squared < 1e-10:
            print("g is in the linear space of phi, extremal datasets don't exist.")
            return None, None
        # Then, compute the direction of maximal change.
        direction = g_perp(self.records)
        max_df = self._maximal_dataset(direction, num_repeats, scaling)
        min_df = self._maximal_dataset(-direction, num_repeats, scaling)
        return min_df, max_df

    def _maximal_dataset(self, direction, num_repeats, scaling):
        # First, determine how to compute the counts.
        if self.use_lp:
            # Solve the linear problem exactly!
            solution = scipy.optimize.linprog(
                -direction,
                A_eq=[p(self.records) for p in self.phi],
                b_eq=np.zeros((len(self.phi),)),
                bounds=[-1 / self.subsample_size, 1 - 1 / self.subsample_size],
            )
            if solution.status != 0:
                self.callback = solution
                raise Exception(
                    "Linear program could not be solved. See self.callback for more information."
                )
            distribution = 1 / self.subsample_size + solution.x
            # Then, scale this distribution to get counts.
            total_size = num_repeats * self.subsample_size
            shifted_counts = np.round(distribution * total_size)
        else:
            # Generate a "neutral" dataset, that contains the same records repeated
            # a large number of times.
            counts = np.full((len(self.records),), num_repeats)
            # Then, move the count of record x by alpha * direction * num_repeats.
            # This value is the maximum alpha (scaling) by which we
            alpha = num_repeats / np.abs(direction.min())
            # Shift the counts with this scaling.
            shifted_counts = counts + np.round(alpha * scaling * direction)
        # Construct the dataset from these counts.
        maximal_dataset = []
        for c, r in zip(shifted_counts, self.records):
            maximal_dataset += [np.tile(r, (int(c), 1))]
        return np.concatenate(maximal_dataset)

    def audit(
        self, generator, g, num_samples=10, num_repeats=100, iterator=lambda i: i
    ):
        # Generate the min and max datasets.
        min_df, max_df = self.generate_extremal_datasets(g, num_repeats)
        if min_df is None or max_df is None:
            print("Can't audit g: this function is uniquely defined by phi.")
            return None
        # Generate synthetic samples for each.
        results = []
        for dataset in [min_df, max_df]:
            g_dataset = g(dataset).mean()
            for _ in iterator(range(num_samples)):
                synth_data = generator(dataset)
                results.append((g_dataset, g(synth_data).mean()))
        return results
