"""
orbit subcount stuff for G = Z_2^3

G = Z_2^3 acts on finite set X, |X| = n
orbits look like G/H for subgroups H
since G = F_2^3 as a vector space, subgroups = subspaces:
    order 8:  G itself           (1,  dim 3)
    order 4:  7 planes H_j       (7,  dim 2)
    order 2:  7 lines  K_m       (7,  dim 1)
    order 1:  trivial {0}        (1,  dim 0)

so orbit sizes are |G/H| = 8/|H|:
    size 1:  stab = G        -> a_1 copies    (fixed points)
    size 2:  stab = H_j      -> alpha^(2)_j   (7 types, j=1..7)
    size 4:  stab = K_m      -> alpha^(4)_m   (7 types, m=1..7)
    size 8:  stab = {0}      -> a_8 copies     (free orbits)

we know a_1, a_2 = sum alpha^(2)_j, a_4 = sum alpha^(4)_m, a_8.
the 14 unknowns are alpha^(2)_1,...,alpha^(2)_7, alpha^(4)_1,...,alpha^(4)_7

three families of equations in 14 unknowns:
  (I)   t_i  = n - M_{K_i}   : transposition count for each e_i  (7 eqs)
  (II)  c_j  = M_{H_j} - ... : reduced orbit count for each H_j  (7 eqs)
  (III) f_j  = # size-4 orbits in H_j-restricted action           (7 eqs)

this gives us:
  rank([T; M])  = 7   -> (I) and (II) alone underdetermined
  rank([M; F])  = 14  -> (II) and (III) together uniquely determine everything

the null space of [T; M] is delta^(2) = -A * delta^(4), so transposition
counts are redundant. the restricted size-4 orbit data breaks the degeneracy.
"""

from sage.all import GF, VectorSpace, matrix, QQ, identity_matrix


# ---- group setup ----


def build_group():
    """
    build the group G = Z_2^3 and relevant information


    returns
    -------
    elts : list of 7 nontrivial elements ordered by Sage
    functionals : list of 7 nontrivial elements ordered by Sage
    subs_H : list of lists of indices corresponding to elements in a subgroup
        subs_H[j] = [m : e_m in H_j]
    """
    V = VectorSpace(GF(2), 3)
    elts = [v for v in V if not v.is_zero()]

    functionals = [v for v in V if not v.is_zero()]

    subs_H = []
    for f in functionals:
        H = [m for m, e in enumerate(elts) if f.dot_product(e) == 0]
        subs_H.append(H)

    return elts, functionals, subs_H


# ---- point-hyperplane incidence ----


def point_hyperplane_incidence(subs_H):
    """
    7x7 incidence matrix between nonzero points (1-subspaces) and
    hyperplanes ((n-1)-subspaces) of F_2^3:

        A[j, m] = 1 if e_m in H_j, else 0.

    properties:
      - row sum = col sum = 2^(n-1) - 1 = 3
          (hyperplane contains 2^(n-1) - 1 nonzero vectors;
           by duality each nonzero vector lies in 2^(n-1) - 1 hyperplanes)
      - det(A) != 0, so A is invertible over Q
          (this is what makes the final system solvable)

    small-case identity (n = 3 only, just sanity check):
      - A^T A = 2I + J
      - det(A) = -24

    parameters
    ----------
    subs_H : list of 7 lists of indices
        from build_group(), which elements belong to which H_j.

    returns
    -------
    A : 7x7 matrix over QQ
        the point-hyperplane incidence matrix.
    """
    return matrix(QQ, 7, 7, lambda j, m: 1 if m in subs_H[j] else 0)


# ---- block (I): transposition counts ----


def transposition_matrix(A):
    """
    7x14 matrix T for transposition equations.

    t_i = # of 2-cycles when e_i acts on X

    contributions per orbit type:
      size 1 (stab G): e_i fixes the point -> 0
      size 2 (stab H_j):
        e_i in H_j -> fixes both -> 0
        e_i not in H_j -> swaps them -> 1
        coeff: 1 - A[j, i]
      size 4 (stab K_m):
        m = i -> fixes all 4 -> 0
        m != i -> two transpositions -> 2
      size 8 (stab {0}): 4 transpositions -> absorbed into constant

    so: t_i = sum_j (1-A[j,i]) alpha^(2)_j + sum_{m!=i} 2 alpha^(4)_m + 4 a_8

    parameters
    ----------
    A : 7x7 matrix over QQ
        point-hyperplane incidence matrix from point_hyperplane_incidence().

    returns
    -------
    T : 7x14 matrix over QQ
        columns = [alpha^(2)_1..7 | alpha^(4)_1..7].
    """

    def entry(i, col):
        if col < 7:
            return 1 - A[col, i]
        else:
            m = col - 7
            return 0 if m == i else 2

    return matrix(QQ, 7, 14, entry)


# ---- block (II): order-4 subgroup orbit counts ----


def subgroup_orbit_matrix(A):
    """
    7x14 matrix M for the H_j orbit count equations.

    M_{H_j} = # of H_j-orbits. each G-orbit type contributes:
      size 1: 1 (fixed)
      size 2, stab H_k:
        k = j -> H_j fixes both -> 2
        k != j -> H_j acts transitively -> 1
      size 4, stab K_m:
        e_m in H_j -> H_j-orbits of size 2 -> contributes 2
        e_m not in H_j -> H_j acts freely -> contributes 1
      size 8: H_j acts freely, 8/4 = 2 orbits

    after simplifying and pulling out knowns:
      c_j = alpha^(2)_j + (A * alpha^(4))_j

    so left block = I, right block = A

    parameters
    ----------
    A : 7x7 matrix over QQ
        point-hyperplane incidence matrix.

    returns
    -------
    M : 7x14 matrix over QQ
        columns = [alpha^(2)_1..7 | alpha^(4)_1..7].
    """

    def entry(j, col):
        if col < 7:
            return 1 if col == j else 0
        else:
            m = col - 7
            return A[j, m]

    return matrix(QQ, 7, 14, entry)


# ---- block (III): restricted size-4 orbit counts ----


def restricted_4orbit_matrix(A):
    """
    7x14 matrix F for the H_j-restricted size-4 orbit equations.

    restrict G-action to H_j, count size-4 H_j-orbits.
    this is the key thing that breaks the degeneracy.

    contributions:
      size 1/2 G-orbits: can't produce size-4 H_j-orbits -> 0
      size 4 (stab K_m):
        e_m in H_j -> H_j-orbits have size 2 -> 0
        e_m not in H_j -> H_j acts freely -> 1 size-4 orbit
      size 8: H_j acts freely -> 2 size-4 orbits (goes to constant)

    so f_j = sum_{m : e_m not in H_j} alpha^(4)_m = ((J-A) alpha^(4))_j

    left block = 0, right block = J - A (complementary incidence:
    point NOT in hyperplane)

    parameters
    ----------
    A : 7x7 matrix over QQ
        point-hyperplane incidence matrix.

    returns
    -------
    F : 7x14 matrix over QQ
        columns = [alpha^(2)_1..7 | alpha^(4)_1..7].
    """

    def entry(j, col):
        if col < 7:
            return 0
        else:
            m = col - 7
            return 1 - A[j, m]

    return matrix(QQ, 7, 14, entry)


def print_group_data(elts, functionals, subs_H):
    print("GROUP DATA: G = Z_2^3")

    print("\nNonzero elements e_1, ..., e_7:")
    for i, e in enumerate(elts):
        print(f"  e_{i+1} = {tuple(e)}")

    print("\nOrder-4 subgroups H_j (nontrivial elements):")
    for j in range(7):
        members = ", ".join(f"e_{m+1}" for m in subs_H[j])
        coeffs = [int(x) for x in functionals[j]]
        terms = "+".join(f"x_{k+1}" for k in range(3) if coeffs[k])
        print(f"  H_{j+1} = ker({terms}) : {{ {members} }}")


def main():
    elts, functionals, subs_H = build_group()
    A = point_hyperplane_incidence(subs_H)

    print_group_data(elts, functionals, subs_H)

    print("\nPoint-hyperplane incidence matrix A:")
    print(A)
    print(f"\ndet(A) = {A.det()}")

    # build the three 7x14 equation blocks
    # columns = [alpha^(2)_1..7 | alpha^(4)_1..7]
    T = transposition_matrix(A)
    M = subgroup_orbit_matrix(A)
    F = restricted_4orbit_matrix(A)

    print("COEFFICIENT MATRICES")
    print("columns = [alpha^(2)_1..7 | alpha^(4)_1..7]")

    print("\n(I) Transposition matrix T  (t_i = T[i,:]*x + 4*a_8):")
    print(T)

    print("\n(II) Subgroup orbit matrix M  (c_j = M[j,:]*x):")
    print(M)

    print("\n(III) Restricted 4-orbit matrix F  (f_j - 2*a_8 = F[j,:]*x):")
    print(F)

    # stack different combos and check rank
    # 14 unknowns so need rank 14 for unique solution
    #
    # [T; M] has rank 7 because the transposition eqs are redundant
    # (burnside for H_j gives 4*M_{H_j} = sum_{g in H_j} |X^g|)
    # [M; F] has rank 14 -- the restricted data gives the extra 7 eqs we need
    TM = T.stack(M)
    MF = M.stack(F)
    TF = T.stack(F)
    TMF = TM.stack(F)

    print("RANK ANALYSIS")
    print("=" * 60)
    print(f"  rank(T)       = {T.rank()}")
    print(f"  rank(M)       = {M.rank()}")
    print(f"  rank(F)       = {F.rank()}")
    print(f"  rank([T; M])  = {TM.rank()}")  # 7 -- T is redundant
    print(f"  rank([M; F])  = {MF.rank()}")  # 14 -- full rank!
    print(f"  rank([T; F])  = {TF.rank()}")  # 14
    print(f"  rank([T;M;F]) = {TMF.rank()}")  # 14
    print("\n  unknowns = 14, required rank = 14")

    # null space of [T; M]
    # each null vector: delta^(2) = -A * delta^(4), with sum(delta^(4)) = 0
    # i.e. shifting alpha^(4)_m by +1 and alpha^(4)_{m'} by -1
    # can be compensated by adjusting alpha^(2). only block III detects this.
    ns = TM.right_kernel().basis()
    print("\n" + "=" * 60)
    print(f"NULL SPACE of [T; M]  (dim = {len(ns)})")
    print("These are the ambiguities that block (III) resolves.")
    print("=" * 60)
    for k, v in enumerate(ns):
        a2_part = list(v[:7])
        a4_part = list(v[7:])
        print(f"  v_{k+1}: alpha^(2) = {a2_part},  alpha^(4) = {a4_part}")

    # closed-form solution from blocks (II) + (III)
    #
    # from (II):  c = alpha^(2) + A alpha^(4)
    # from (III): f = (J - A) alpha^(4)
    # adding: c + f = alpha^(2) + J alpha^(4) = alpha^(2) + a_4
    # (this is the identity M + F = [I_7 | J_7])
    #
    # so:
    #   alpha^(2)_j = c_j + f_j - a_4
    #   A alpha^(4) = c - alpha^(2) = a_4 - f
    #   alpha^(4) = A^{-1} (a_4 * 1 - f)
    #
    # works because det(A) = -24 != 0
    Ainv = A.inverse()

    print("\n" + "=" * 60)
    print("SOLUTION  (from blocks II + III)")
    print("=" * 60)
    print(
        """
Define:
  c_j = M_{H_j} - a_1 - a_2 - a_4 - 2*a_8
  f_j = (# size-4 orbits in H_j-restricted action) - 2*a_8

Then:
  alpha^(2)_j = c_j + f_j - a_4
  alpha^(4)   = A^{-1} * (a_4 * 1 - f)
"""
    )

    print("A^{-1}:")
    print(Ainv)

    print("\ncheck: 6 * A^{-1}  (cleared denominators):")
    print((6 * Ainv).change_ring(QQ))

    # sanity checks
    I7 = identity_matrix(QQ, 7)
    J7 = matrix(QQ, 7, 7, lambda i, j: 1)

    # small-case identity: A^T A = 2I + J (n=3 only; any two distinct
    # points lie in exactly 2^(n-2) = 2 common hyperplanes, and each
    # point lies in 2^(n-1) - 1 = 3 hyperplanes)
    assert A.transpose() * A == 2 * I7 + J7, "A^T A != 2I + J"
    print("Success")

    # M + F = [I_7 | J_7] -- this is why the closed form works
    MpF = M + F
    print("\ncheck: M + F  (should be [I_7 | J_7]):")
    print(MpF)
    assert MpF[:, :7] == I7, "M+F left block != I"
    assert MpF[:, 7:] == J7, "M+F right block != J"
    print("Success")

    # Final matrix
    S = M.stack(F)

    # Its inverse
    Sinv = S.inverse()

    print("\nfinal matrix:")
    print("S =\n", S, sep="")

    print("\ncheck: S is invertible ")
    print("S^(-1) =\n", Sinv, sep="")
    print("Success")


if __name__ == "__main__":
    main()
