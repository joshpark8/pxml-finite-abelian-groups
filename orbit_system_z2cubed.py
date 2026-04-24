"""
orbit subcount stuff for G = Z_2^3

G acts on finite set X, |X| = n.
orbits are G-isomorphic to G/H for subgroups H <= G, with |orbit| = [G:H].

subgroup lattice of Z_2^3:
    order 8:  G                 (1 subgroup)
    order 4:  H_1, ..., H_7     (7 index-2 subgroups)
    order 2:  K_1, ..., K_7     (7 order-2 subgroups)
    order 1:  {0}               (1 subgroup)

orbit sizes = [G:H], so X decomposes as:
    size 1:  stab = G        -> a_1 copies    (fixed points)
    size 2:  stab = H_j      -> alpha^(2)_j   (7 types, j=1..7)
    size 4:  stab = K_m      -> alpha^(4)_m   (7 types, m=1..7)
    size 8:  stab = {0}      -> a_8 copies    (free orbits)

we know the totals a_1, a_2 = sum alpha^(2)_j, a_4 = sum alpha^(4)_m, a_8.
the 14 unknowns are alpha^(2)_1,...,alpha^(2)_7, alpha^(4)_1,...,alpha^(4)_7.

three families of equations:
  (I)   t_i = # 2-cycles of g_i acting on X, one equation per
        nontrivial g_i in G.                                    (7 eqs)
        NOTE: Z_2^3-specific. every nontrivial g here has order 2, so
        cycle structure collapses to fixed points + transpositions.
        for general abelian G, elements have higher order and this
        block must record full cycle structure per element.
  (II)  c_j = reduced H_j-orbit count on X                      (7 eqs)
  (III) f_j = # size-2 H_j-orbits coming from size-4 G-orbits   (7 eqs)

rank summary:
  rank([T; M])  = 7   -> (I) and (II) alone underdetermined
  rank([M; F])  = 14  -> (II) and (III) together uniquely determine everything

null space of [T; M] is delta^(2) = -A * delta^(4), so cycle counts are
redundant given (II). the restricted orbit data in (III) breaks the
degeneracy.
"""

from sage.all import GF, VectorSpace, matrix, QQ, identity_matrix


# ---- group setup ----


def build_group():
    """
    build G = Z_2^3 (as an additive abelian group) and enumerate:
      - its nontrivial elements
      - its index-2 subgroups

    returns
    -------
    elts : list of 7 nontrivial elements of G, with stable indexing.
    subs_H : list of 7 index-2 subgroups. each is a list of indices
        m with elts[m] in that subgroup, so subs_H[j] = [m : e_m in H_j].
    """
    # VectorSpace(GF(2), 3) is just a convenient realization of Z_2^3;
    # only the additive group structure is used below.
    # for a general finite abelian group, use AdditiveAbelianGroup([...])
    # and G.subgroups() filtered by index.
    V = VectorSpace(GF(2), 3)
    elts = [v for v in V if not v.is_zero()]

    subs_H = []
    for H in V.subspaces(2):
        members = [m for m, e in enumerate(elts) if e in H]
        subs_H.append(members)

    return elts, subs_H


# ---- subgroup incidence ----


def incidence_matrix(subs_H):
    """
    7x7 incidence matrix between order-2 subgroups <e_m> and index-2
    subgroups H_j:

        A[j, m] = 1 if e_m in H_j (equivalently <e_m> <= H_j), else 0.

    rows = index-2 subgroups, cols = order-2 subgroups. these have equal
    cardinalities here (both 7) because the subgroup lattice of a finite
    abelian group is self-dual: # subgroups of order d = # subgroups of
    index d.

    properties used below:
      - row sum = col sum = 3 in Z_2^3.
      - det(A) != 0, so A is invertible over Q; this is what makes the
        final system solvable.

    NOTE: the identity A^T A = 2I + J holds for Z_2^3 but does not
    generalize. for general abelian G, the incidence matrix between
    order-d and index-d subgroups will be rectangular in general, and
    only square when sizes happen to coincide.

    parameters
    ----------
    subs_H : list of 7 lists of indices, from build_group().

    returns
    -------
    A : 7x7 matrix over QQ.
    """
    return matrix(QQ, 7, 7, lambda j, m: 1 if m in subs_H[j] else 0)


# ---- block (I): transposition counts ----


def transposition_matrix(A):
    """
    7x14 matrix T for the cycle-count equations of block (I).

    for each nontrivial g_i in G, t_i = # 2-cycles of g_i acting on X.

    NOTE: Z_2^3-specific. every nontrivial g in Z_2^3 has order 2, so
    its cycle structure on X is (fixed points) + (transpositions). for
    general abelian G, block (I) must record full cycle structure per
    element (e.g., 3-cycles for order-3 elements, etc.).

    contributions per orbit type, under action of e_i:
      size 1 (stab G): e_i fixes the point -> 0
      size 2 (stab H_j):
        e_i in H_j -> fixes both -> 0
        e_i not in H_j -> swaps them -> 1
        coeff: 1 - A[j, i]
      size 4 (stab K_m):
        m = i -> e_i fixes all 4 -> 0
        m != i -> e_i acts as two transpositions -> 2
      size 8 (stab {0}): 4 transpositions -> absorbed into constant

    so: t_i = sum_j (1-A[j,i]) alpha^(2)_j + sum_{m!=i} 2 alpha^(4)_m + 4 a_8

    parameters
    ----------
    A : 7x7 matrix over QQ
        incidence matrix from incidence_matrix().

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


# ---- block (II): index-2 subgroup orbit counts ----


def subgroup_orbit_matrix(A):
    """
    7x14 matrix M for the H_j-orbit count equations of block (II).

    M_{H_j} = # of H_j-orbits on X. each G-orbit type contributes:
      size 1: 1 (H_j fixes it)
      size 2, stab H_k:
        k = j -> H_j fixes both points -> 2 orbits
        k != j -> H_j acts transitively -> 1 orbit
      size 4, stab K_m:
        e_m in H_j -> H_j-orbit splits into 2 orbits of size 2
        e_m not in H_j -> H_j acts freely -> 1 orbit
      size 8: H_j acts freely, 8/|H_j| = 2 orbits

    after simplifying and pulling out knowns:
      c_j = alpha^(2)_j + (A * alpha^(4))_j

    so left block = I, right block = A

    parameters
    ----------
    A : 7x7 matrix over QQ

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
    7x14 matrix F for the H_j-restricted size-4 orbit equations, block (III).

    restrict the G-action to H_j and count size-4 H_j-orbits on X.
    this is the data that breaks the degeneracy of (I)+(II).

    contributions:
      size 1/2 G-orbits: cannot produce size-4 H_j-orbits -> 0
      size 4 (stab K_m):
        e_m in H_j -> H_j-orbits have size 2 -> 0
        e_m not in H_j -> H_j acts freely -> 1 size-4 orbit
      size 8: H_j acts freely -> 2 size-4 orbits (absorbed into constant)

    so f_j = sum_{m : e_m not in H_j} alpha^(4)_m = ((J-A) alpha^(4))_j

    left block = 0, right block = J - A (complementary incidence:
    <e_m> NOT contained in H_j).

    parameters
    ----------
    A : 7x7 matrix over QQ
        incidence matrix.

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


def print_group_data(elts, subs_H):
    print("GROUP DATA: G = Z_2^3")

    print("\nNontrivial elements e_1, ..., e_7:")
    for i, e in enumerate(elts):
        print(f"  e_{i+1} = {tuple(e)}")

    print("\nIndex-2 subgroups H_j (listed by their nontrivial elements):")
    for j in range(7):
        members = ", ".join(f"e_{m+1}" for m in subs_H[j])
        print(f"  H_{j+1} = {{ 0, {members} }}")


def main():
    elts, subs_H = build_group()
    A = incidence_matrix(subs_H)

    print_group_data(elts, subs_H)

    print("\nIncidence matrix A (rows = H_j, cols = <e_m>):")
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

    # Z_2^3-specific sanity check (does NOT generalize):
    # any two distinct order-2 subgroups lie together in exactly 2 index-2
    # subgroups, and each order-2 subgroup lies in 3 index-2 subgroups.
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
