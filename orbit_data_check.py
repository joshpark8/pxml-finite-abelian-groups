"""Do the subgroup orbit counts and orbit multiplicities determine a G-set?

An independent check of the setup in "Subgroup orbit count identity" and
"Combinatorial Uniqueness of Z_2^3-sets".  Everything here is computed by
building G-sets as explicit permutation actions and counting orbits by hand --
none of the mark machinery in counterexample.py is used, so this is a genuine
second opinion rather than a restatement.

Notation follows those notes:

    a_d       d-orbit multiplicity -- the number of size-d orbits in G\\X
    M_H       subgroup orbit count |H\\X|
    alpha_H   the number of orbits whose stabilizer is exactly H
    S_d(X)    the d-orbit submultiplicities, i.e. (alpha_H) over [G:H] = d

The question under test: if M_H(X) = M_H(Y) for every H <= G and a_d(X) = a_d(Y)
for every d, must X and Y be isomorphic -- equivalently, must S_d(X) = S_d(Y)?
"""

N = 8                                   # |G| for G = Z_2^3, elements as bitstrings
bits = lambda x: format(x, "03b")


def span(gens):
    S = {0}
    for g in gens:
        S |= {s ^ g for s in S}
    return frozenset(S)


# labels exactly as in "Combinatorial Uniqueness of Z_2^3-sets"
H = {1: span([0b001]), 2: span([0b010]), 3: span([0b011]), 4: span([0b100]),
     5: span([0b101]), 6: span([0b110]), 7: span([0b111])}
K = {1: span([0b001, 0b010]), 2: span([0b001, 0b100]), 3: span([0b001, 0b110]),
     4: span([0b010, 0b100]), 5: span([0b010, 0b101]), 6: span([0b011, 0b100]),
     7: span([0b011, 0b101])}
TRIV, FULL = span([]), span([1, 2, 4])

SUBGROUPS = [TRIV] + [H[i] for i in range(1, 8)] + [K[j] for j in range(1, 8)] + [FULL]
NAMES = {TRIV: "1", FULL: "G"}
NAMES.update({S: "H%d" % i for i, S in H.items()})
NAMES.update({S: "K%d" % j for j, S in K.items()})


# ----------------------------------------------------------- explicit G-sets

def gset(stabilizers):
    """The G-set  |_| G/S  as an explicit point set; a point is (component, coset)."""
    pts = []
    for i, S in enumerate(stabilizers):
        for c in {frozenset(x ^ s for s in S) for x in range(N)}:
            pts.append((i, c))
    return pts


def act(g, pt):
    i, c = pt
    return (i, frozenset(x ^ g for x in c))


def orbits(pts, sub):
    """Orbits of the subgroup sub on pts, found by closing under the action."""
    seen, out = set(), []
    for p in pts:
        if p in seen:
            continue
        orb = {act(g, p) for g in sub}
        seen |= orb
        out.append(orb)
    return out


def M(pts, sub):
    """Subgroup orbit count M_H = |H\\X|, by direct orbit counting."""
    return len(orbits(pts, sub))


def stabilizer(pt):
    return frozenset(g for g in range(N) if act(g, pt) == pt)


def a_d(pts):
    """d-orbit multiplicities of the G-action, as {d: a_d}."""
    counts = {}
    for orb in orbits(pts, FULL):
        counts[len(orb)] = counts.get(len(orb), 0) + 1
    return counts


def alpha(pts):
    """alpha_H for every H: the number of G-orbits with stabilizer exactly H."""
    out = {S: 0 for S in SUBGROUPS}
    for orb in orbits(pts, FULL):
        out[stabilizer(next(iter(orb)))] += 1
    return out


def S_d(pts, d):
    """The d-orbit submultiplicities: alpha_H over those H with [G:H] = d."""
    al = alpha(pts)
    return {NAMES[S]: al[S] for S in SUBGROUPS if N // len(S) == d}


# ------------------------------------------- checks on the identities themselves

def check_identities(pts):
    """Theorem 1.1 and Corollary 1.2.1, verified against brute-force counts."""
    al, mult = alpha(pts), a_d(pts)
    ok = True
    for d in sorted(mult):
        lhs = mult[d]
        rhs = sum(al[S] for S in SUBGROUPS if N // len(S) == d)
        ok &= lhs == rhs
        print("    Thm 1.1   a_%d = %d = sum of alpha_H over [G:H]=%d (%d)  %s"
              % (d, lhs, d, rhs, "ok" if lhs == rhs else "FAIL"))
    for sub in SUBGROUPS:
        lhs = M(pts, sub)
        rhs = sum(al[Kk] * (N // len(span(list(sub) + list(Kk)))) for Kk in SUBGROUPS)
        ok &= lhs == rhs
    print("    Cor 1.2.1 M_H = sum_K alpha_K [G:HK] for all 16 subgroups:  %s"
          % ("ok" if ok else "FAIL"))
    return ok


# ---------------------------------------------------------------------- main

def compare(Xs, Ys):
    px, py = gset(Xs), gset(Ys)
    nx = " + ".join("G/" + NAMES[S] for S in Xs)
    ny = " + ".join("G/" + NAMES[S] for S in Ys)
    print("X = %s      |X| = %d" % (nx, len(px)))
    print("Y = %s      |Y| = %d" % (ny, len(py)))

    print("\nidentity checks (validates this script against the notes):")
    print("  X:"); check_identities(px)
    print("  Y:"); check_identities(py)

    print("\nsubgroup orbit counts M_H (direct orbit counting):")
    print("   %-4s %8s %8s" % ("H", "M_H(X)", "M_H(Y)"))
    same_M = True
    for sub in SUBGROUPS:
        mx, my = M(px, sub), M(py, sub)
        same_M &= mx == my
        print("   %-4s %8d %8d %s" % (NAMES[sub], mx, my, "" if mx == my else "  <-- differs"))

    ax, ay = a_d(px), a_d(py)
    print("\norbit multiplicities a_d:")
    for d in sorted(set(ax) | set(ay)):
        print("   a_%-2d %8d %8d" % (d, ax.get(d, 0), ay.get(d, 0)))
    same_a = ax == ay

    print("\norbit submultiplicities S_d:")
    same_S = True
    for d in sorted(set(ax) | set(ay)):
        sx, sy = S_d(px, d), S_d(py, d)
        same_S &= sx == sy
        print("   S_%-2d(X) = %s" % (d, {k: v for k, v in sx.items() if v}))
        print("   S_%-2d(Y) = %s%s" % (d, {k: v for k, v in sy.items() if v},
                                       "" if sx == sy else "     <-- differs"))

    print("\n  M_H(X) = M_H(Y) for all H <= G : %s" % same_M)
    print("  a_d(X) = a_d(Y) for all d      : %s" % same_a)
    print("  S_d(X) = S_d(Y) for all d      : %s" % same_S)
    print("  => X %s Y" % ("~=" if same_S else "NOT ISOMORPHIC to"))
    if same_M and same_a and not same_S:
        print("\n  X and Y agree on every subgroup orbit count and every orbit")
        print("  multiplicity, yet are non-isomorphic.")


# ------------------------------- the linear system of "Combinatorial Uniqueness"

def fix(pts, g):
    return sum(1 for p in pts if act(g, p) == p)


def fix_sub(pts, sub):
    return sum(1 for p in pts if all(act(g, p) == p for g in sub))


def uniqueness_system(pts):
    """Recompute tau, beta, x and M_{alpha_K} alpha_K as defined in the notes.

    tau, beta and x depend only on the M_H data; alpha_K does not.  So if two
    G-sets share their M_H and a_d but differ in alpha_K, the asserted identity
    M_{alpha_K} alpha_K = x cannot hold for both of them."""
    n = len(pts)
    al = alpha(pts)
    a8 = al[TRIV]
    tau = [(n - fix(pts, i)) // 2 for i in range(1, 8)]           # tau_1 .. tau_7
    n_K = [sum((n - fix(pts, g)) // 2 for g in K[j] if g != 0) for j in range(1, 8)]
    beta = [(n_K[j - 1] - (n - fix_sub(pts, K[j]))) // 2 - 2 * a8 for j in range(1, 8)]
    MK = [[1 if H[i] <= K[j] else 0 for j in range(1, 8)] for i in range(1, 8)]
    x = [tau[i] - 4 * a8 - sum(MK[i][j] * beta[j] for j in range(7)) for i in range(7)]
    MaK = [[1 if not H[i] <= K[j] else 0 for j in range(1, 8)] for i in range(1, 8)]
    aK = [al[K[j]] for j in range(1, 8)]
    lhs = [sum(MaK[i][j] * aK[j] for j in range(7)) for i in range(7)]
    return dict(a8=a8, tau=tau, beta=beta, x=x, aK=aK, lhs=lhs, holds=lhs == x)


def compare_systems(Xs, Ys):
    px, py = gset(Xs), gset(Ys)
    sx, sy = uniqueness_system(px), uniqueness_system(py)
    print("\nthe linear system of \"Combinatorial Uniqueness of Z_2^3-sets\":")
    for key in ("tau", "beta", "x"):
        print("   %-5s X: %-28s Y: %-28s %s"
              % (key, sx[key], sy[key], "same" if sx[key] == sy[key] else "DIFFER"))
    print("   %-5s X: %-28s Y: %-28s %s"
          % ("aK", sx["aK"], sy["aK"], "same" if sx["aK"] == sy["aK"] else "DIFFER"))
    print("\n   Thm 2.2 asserts  M_{alpha_K} alpha_K = x :")
    for lbl, s in (("X", sx), ("Y", sy)):
        print("     %s:  M_{alpha_K} alpha_K = %-24s x = %-24s %s"
              % (lbl, s["lhs"], s["x"], "holds" if s["holds"] else "FAILS"))
    if sx["holds"] and sy["holds"] and sx["x"] != sy["x"]:
        print("\n   Thm 2.2 holds for both, so the break is upstream: x differs even")
        print("   though tau (which the M_H data fixes) does not.  x differs because")
        print("   beta does, and beta is built by Lemma 2.1 out of |X^{K_i}|:")
        print("\n   %-4s %9s %9s   %9s %9s" % ("K", "M_K(X)", "M_K(Y)", "|X^K|", "|Y^K|"))
        for j in range(1, 8):
            mx, my = M(px, K[j]), M(py, K[j])
            fx, fy = fix_sub(px, K[j]), fix_sub(py, K[j])
            print("   %-4s %9d %9d   %9d %9d %s"
                  % ("K%d" % j, mx, my, fx, fy, "" if fx == fy else "  <-- differs"))
        print("\n   The subgroup orbit counts M_K agree everywhere, but the fixed-point")
        print("   counts |X^K| do not.  So |X^{K_i}| is NOT recoverable from {M_H} and")
        print("   {a_d}, and Lemma 2.1 is using data outside its stated hypotheses.")
        print("   (X^K = intersection of X^g over generators g of K; the size of an")
        print("   intersection is not a function of the sizes of the pieces.)")


def submultiplicity_reading(Xs, Ys):
    """Definition 1.3 written with set braces is ambiguous, and it matters here."""
    px, py = gset(Xs), gset(Ys)
    print("\nreading of Definition 1.3:")
    for d in sorted(set(a_d(px)) | set(a_d(py))):
        vx, vy = S_d(px, d), S_d(py, d)
        mx, my = sorted(vx.values(), reverse=True), sorted(vy.values(), reverse=True)
        print("   d = %d" % d)
        print("     as a function H |-> alpha_H : %s" % ("equal" if vx == vy else "DIFFERENT"))
        print("     as a multiset of values     : %s   %s / %s"
              % ("equal" if mx == my else "different", mx, my))


if __name__ == "__main__":
    X, Y = [K[1], K[2], H[7]], [K[5], K[6], H[1]]
    compare(X, Y)
    submultiplicity_reading(X, Y)
    compare_systems(X, Y)
