# Algorithms and their mathematics

This note records the mathematics behind each kernel in the native backend so
that a reader can check an implementation against the formula it is supposed
to compute. `docs/native-lu.md` describes how the engines are selected and
what the `stats` fields mean; this file describes what they compute.

Notation: $A \in \mathbb{R}^{n \times n}$ is the input, $a_{ij}$ its entries,
$\mathrm{Struct}(v)$ the index set of nonzeros of a sparse vector or matrix
column, $|M|$ the entrywise absolute value of a matrix, and
$\mathrm{eps} = 2^{-52}$ the double-precision machine epsilon
(`np.finfo(np.float64).eps`) used wherever a tolerance is quoted below.

## 1. Stored factorization and solves

Every native factor stores

$$P_r D_r A D_c P_c = L U,$$

with permutation matrices $P_r, P_c$, positive diagonal scalings $D_r, D_c$,
unit lower triangular $L$ and upper triangular $U$. In the code `perm_r[k]`
and `perm_c[k]` are the original row and column at factor position $k$, and
`row_scale`, `col_scale` hold the diagonals of $D_r$, $D_c$ indexed by original
row/column.

For the symmetric engines the stored object is $P^T S A S P = L D L^T$ with a
single symmetric permutation $P$ (`perm_r == perm_c`) and a single scaling
$S$; it is exposed through the same contract with $U = D L^T$ and
$D_r = D_c = S$.

### Solving $Ax = b$

Writing $y = P_r D_r b$ (i.e. $y_k = d_{r,\,\pi_r(k)}\, b_{\pi_r(k)}$ with
$\pi_r$ = `perm_r`), solve $L z = y$ by forward substitution and $U w = z$ by
back substitution, then scatter $x_{\pi_c(k)} = d_{c,\,\pi_c(k)}\, w_k$.

### Solving $A^T x = b$

From $(P_r D_r A D_c P_c)^T = P_c^T D_c A^T D_r P_r^T = U^T L^T$: gather
$y_k = d_{c,\,\pi_c(k)}\, b_{\pi_c(k)}$, solve $U^T z = y$ (lower triangular,
forward), then $L^T w = z$ (upper triangular, backward), and scatter
$x_{\pi_r(k)} = d_{r,\,\pi_r(k)}\, w_k$. `trans="H"` equals `"T"` for real
factors.

For the LDL^T storage, $U^{-1} = L^{-T} D^{-1}$, so a solve is
$L z = y$, $z \leftarrow D^{-1} z$, $L^T w = z$ in both the normal and the
transposed case.

Right-hand sides are processed in blocks of four columns; blocks are
independent and run in parallel. Within a block the substitution order is
fixed, so results do not depend on the thread count.

## 2. Equilibration

Row scaling first, then column scaling of the row-scaled matrix:

$$d_{r,i} = \frac{1}{\max_j |a_{ij}|}, \qquad
  d_{c,j} = \frac{1}{\max_i |d_{r,i}\, a_{ij}|}.$$

After scaling every column of $D_r A D_c$ has infinity norm exactly $1$ and
every row has infinity norm at most $1$. A zero row or column makes the
corresponding maximum $0$ and is reported as structural singularity before
any elimination; an overflowing reciprocal is reported as a scaling overflow.

The symmetric path uses $S = D_r^{1/2}$ on both sides, $\tilde A = S A S$, so
that $\tilde a_{ij} = \sqrt{d_{r,i}\, d_{r,j}}\, a_{ij}$ keeps $\tilde A$
exactly symmetric and keeps diagonal dominance ratios unchanged.

Equilibration does not change the exact solution: $Ax = b$ iff
$(D_r A D_c)(D_c^{-1} x) = D_r b$.

## 3. Threshold partial pivoting

At elimination step $k$ of the column-permuted, scaled matrix, let $C_k$ be the
rows not yet chosen as pivots with a nonzero in column $k$ (after all updates
from steps $< k$) and let

$$m_k = \max_{r \in C_k} |\tilde a_{rk}|.$$

The preferred row is $p = \pi_c(k)$, the row whose original index equals the
column's original index, so that a symmetric permutation keeps the diagonal
when it is numerically acceptable. The pivot row is

$$\text{pivot}_k = \begin{cases}
  p & \text{if } p \in C_k \text{ and } |\tilde a_{pk}| \ge \tau\, m_k,\\
  \arg\max_{r \in C_k} |\tilde a_{rk}| & \text{otherwise},
\end{cases}$$

with the threshold $\tau$ = `diag_pivot_thresh` $\in [0, 1]$ (default $1$,
which is ordinary partial pivoting; the preferred row then wins only ties).
Ties in the maximum are broken by the smallest current row position, which
makes the choice independent of the thread count. Every multiplier satisfies

$$|l_{rk}| = \frac{|\tilde a_{rk}|}{|\tilde a_{\text{pivot}_k, k}|} \le \frac{1}{\tau}.$$

If $C_k$ is empty or $m_k = 0$ the matrix is exactly singular at column $k$.
If $m_k$ or the pivot is not finite, the factorization is aborted as
non-finite.

## 4. Left-looking CSC LU (Gilbert–Peierls)

The `left-looking-csc` engine computes one column of $L$ and $U$ per step.
Let $\tilde A$ be the scaled, column-permuted matrix and let $L_{(k)}$ denote
the first $k$ columns of $L$ in pivot-row coordinates. Column $k$ solves the
sparse triangular system

$$L_{(k)}\, x = \tilde a_{:,k},$$

restricted to the already pivoted rows, which gives $u_{:,k}$; the remaining
components of $x$ are the candidates for the pivot and, divided by the
pivot, the column $l_{:,k}$:

$$u_{jk} = x_{j}\ (j < k \text{ in pivot order}), \qquad
  u_{kk} = x_{\text{pivot}_k}, \qquad
  l_{rk} = x_r / u_{kk}\ (r \text{ unpivoted}).$$

The nonzero pattern of $x$ is obtained before any arithmetic:

$$\mathrm{Struct}(x) = \mathrm{Reach}_{G(L_{(k)})}\big(\mathrm{Struct}(\tilde a_{:,k})\big),$$

where $G(L_{(k)})$ is the directed graph with an edge $j \to i$ for every
$l_{ij} \ne 0$. A depth-first search from each nonzero of $\tilde a_{:,k}$
returns the reach in topological order; the numeric solve then processes the
reached pivot columns $j$ in that order:

$$x_i \leftarrow x_i - l_{ij}\, x_j \quad \text{for } i \in \mathrm{Struct}(l_{:,j}),\ i \ne j.$$

Gilbert and Peierls showed that with this organisation the total time is
$O(\mathrm{flops}(LU))$ plus the input size, independent of fill, which is
why the engine is used for narrow fronts. The COLAMD "probe" runs this
algorithm with budgets (reach above 192 rows, more than $8192$ updates in one
column, more than $64\,\mathrm{nnz}(A)$ updates in total, or more than
$20\,\mathrm{nnz}(A)$ factor entries) and restarts in the row engine when a
budget is exceeded, so the discarded work is bounded by a constant multiple of
the input size.

## 5. Row-hybrid right-looking LU

The `row-hybrid` engine stores the active submatrix by rows and applies each
pivot's update immediately (right-looking). After choosing pivot row $p$ at
step $k$, with $u_{kj} = \tilde a_{pj}$ for $j \ge k$, every remaining row $r$
with $\tilde a_{rk} \ne 0$ receives

$$l_{rk} = \frac{\tilde a_{rk}}{u_{kk}}, \qquad
  \tilde a_{rj} \leftarrow \tilde a_{rj} - l_{rk}\, u_{kj} \quad (j > k).$$

For a sparse row this is an ordered merge of two column lists; the fill-in of
row $r$ is $\mathrm{Struct}(\tilde a_{r,>k}) \cup \mathrm{Struct}(u_{k,>k})$.
Fill makes rows denser as elimination proceeds, so a row is converted to a
contiguous dense tail once it is dense relative to its own span:

$$\mathrm{nnz}(r) \ge 64 \quad\text{and}\quad
  4\,\mathrm{nnz}(r) \ge \text{(last remaining column)} - \text{(first remaining column)} + 1.$$

The tail covers exactly that span (rounded up to a multiple of 64 columns) and
grows when a pivot row introduces columns beyond it; columns outside the tail
are exact zeros. Measuring density against the row's own span rather than
the global remaining width keeps block-structured matrices (several dense
blocks in one system) on the dense path.

### Blocked dense updates

Dense rows are updated in panels of $w = 32$ pivot columns. Let $P$ be the
panel's column range, $T$ the columns after the panel and $R$ the dense rows
still active at the panel's end. Within the panel every coefficient in
columns of $P$ is kept current, because pivot search needs it: the dense
rows' entries in $P$ are copied into a column-major block $B_{R,P}$, which
is factored like an unblocked dense LU (LAPACK's `dgetf2`). At step $k$ the
block column is scanned for the candidates, divided by the pivot to give
$l_{rk}$, and each later block column $j$ with $u_{kj} \ne 0$ receives the
rank-1 update $b_{rj} \leftarrow b_{rj} - l_{rk} u_{kj}$ over all rows at
once. Rows that are not candidates have $l_{rk} = 0$, and the pivot row's
block row is cleared at its step, so no target list is needed. The updates
to columns in $T$ are deferred and applied once as

$$\tilde A_{R,T} \leftarrow \tilde A_{R,T} - L_{R,P}\, U_{P,T},$$

the standard blocked-LU identity: the rank-1 updates of the $w$ pivots
commute because each term $l_{rk} u_{kj}$ depends only on finalized entries.
When a dense row becomes the pivot row in the middle of a panel its deferred
tail must be complete first, so it is materialized explicitly:

$$u_{kj} = \tilde a_{pj} - \sum_{k' \in P,\ k' < k} l_{p k'}\, u_{k' j}, \qquad j \in T,$$

summing in ascending $k'$. The flush packs the columns of $T$ in which some
$u_{kj} \ne 0$ as contiguous column segments: the nonzero columns of the
dense pivot rows (a column mask) merged with the sparse pivot rows' columns,
with gaps of fewer than 16 columns bridged and wider gaps splitting
segments. A bridged column, where every $u_{kj}$ is zero, changes at most
the sign of a zero entry. The flush is tiled over disjoint blocks of 16
rows by at most 64 columns of one segment; tiles may run on different
threads because they write disjoint entries and each entry subtracts its $w$
terms in ascending pivot order.

## 6. Symmetric $LDL^T$

For an exactly symmetric input the engines compute

$$P^T \tilde A P = L D L^T,$$

with unit lower triangular $L$ and diagonal $D$, pivoting only through the
prescribed symmetric permutation $P$ (the requested column ordering applied to
both sides). Such a factorization exists with all $d_k > 0$ if and only if
$\tilde A$ is symmetric positive definite; the engines therefore accept the
result only when every pivot is positive and finite and otherwise fall back to
pivoted LU of the same matrix. Indefinite matrices are never factored
symmetrically.

Exact symmetry ($a_{ij}$ stored iff $a_{ji}$ stored, with equal values) is
tested in $O(\mathrm{nnz}(A) + n)$ on the sorted CSC input by a transpose
walk: columns $j = 0, 1, \dots$ are visited in order, and each entry
$(i, j)$ must equal the next unconsumed entry of column $i$, which must have
row index $j$. The entries of $A$ that name column $i$ as their transpose
arrive in increasing $j$, the sorted order of column $i$, so a symmetric
matrix passes every step; conversely, a passing walk pairs every entry with a
distinct equal transpose.

### Column recurrence (simplicial kernel)

Working on the permuted matrix $B = P^T \tilde A P$, step $k$ determines row
$k$ of $L$ and $d_k$ from the already computed columns $0..k-1$ by solving the
lower triangular system

$$L_{0:k,\,0:k}\; y = b_{0:k,\,k}, \qquad
  l_{kj} = \frac{y_j}{d_j}\ (j < k), \qquad
  d_k = b_{kk} - \sum_{j<k} l_{kj}\, y_j = b_{kk} - \sum_{j<k} l_{kj}^2\, d_j.$$

Only the nonzero $y_j$ are visited. Their index set is the row subtree
of the elimination tree (Section 7):
$\mathrm{Struct}(l_{k,:}) = \bigcup_{i \in \mathrm{Struct}(b_{0:k,k})} \mathrm{path}(i \to k)$,
the union of the tree paths from each nonzero row index up to node $k$,
visited in topological (leaf-to-root) order. The code accumulates $y$ in a
dense work vector, applies $y_i \leftarrow y_i - l_{ij} y_j$ over the stored
entries of column $j$, then appends $l_{kj} = y_j / d_j$ to column $j$. Both
the `symmetric-ldlt-csc` engine and the first phase of the supernodal engine
produce exactly this $L$ and $D$; they differ only in how the arithmetic is
grouped.

### Supernodal kernel

A fundamental supernode is a maximal run of consecutive columns
$J = \{j, j+1, \dots, j+W-1\}$ such that each column's parent in the
elimination tree is the next column and
$\mathrm{Struct}(l_{:,t+1}) = \mathrm{Struct}(l_{:,t}) \setminus \{t+1\}$
for consecutive $t, t+1 \in J$. Its row set $R$ (height $H = |R|$, the
columns themselves first, then the shared tail) is the pattern of its first
column, so the node's part of $L$ is a dense $H \times W$ block
(trapezoid). The engine assembles that block as a dense front $F$ from the
entries of $B$ and subtracts the contributions of every earlier supernode $Q$
whose rows intersect $J$:

$$F_{R \cap R_Q,\; J \cap R_Q} \;\leftarrow\; F_{R \cap R_Q,\; J \cap R_Q}
  - L_{R \cap R_Q,\,Q}\; D_Q\; L_{J \cap R_Q,\,Q}^{\,T}.$$

The rows of $Q$ that fall inside the target are a contiguous suffix of $Q$'s
row list (a consequence of the elimination tree: once a column of $Q$ has a
nonzero in a row of $J$, all its later rows are in $R$), which is what makes
the operands dense rectangles. Each node's $H \times W$ block is kept, row
major, in one array for all nodes until every node is factored, so
$L_{R \cap R_Q,\,Q}$ is a run of consecutive full rows of $Q$'s finished block
and is read in place; only $D_Q L_{J \cap R_Q,\,Q}^{\,T}$ is packed. The CSC
factor is copied from the blocks once, after the last pivot has been
accepted. For a node of width $W$ and height $H$ the block holds
$HW = \mathrm{nnz} + W(W+1)/2$ entries for its
$\mathrm{nnz} = HW - W(W+1)/2$ strict entries of $L$, and
$HW \le 2\,\mathrm{nnz} + W$ because $H \ge W$; summed over the nodes, the
blocks hold at most $2\,\mathrm{nnz}(L) + n$ entries. Contributors are
visited in increasing node order, and each output entry accumulates its terms
in that order regardless of threading. Each contributor is one rank-$|Q|$
update: an entry forms $s = \sum_{q \in Q} l_{iq}\, (d_q\, l_{jq})$ from zero
in ascending $q$ and subtracts $s$ once (Section 11).

The diagonal block $F_{J,J}$ is then factored without pivoting in panels of
$w = 32$ columns. For a panel $P \subset J$ with the rows $R'$ below it:

$$F_{P,P} = L_{P,P} D_P L_{P,P}^T, \qquad
  L_{R',P} = F_{R',P}\, L_{P,P}^{-T}\, D_P^{-1}, \qquad
  F_{R',J'} \leftarrow F_{R',J'} - L_{R',P}\, D_P\, L_{J',P}^{\,T}$$

for the remaining node columns $J'$ (lower triangle only) and the tail rows.
The triangular solve is the recurrence
$l_{ip} = (f_{ip} - \sum_{k<p} l_{ik}\, d_k\, l_{pk}) / d_p$, the same as the
simplicial kernel restricted to the node. The code evaluates it in AXPY form
with $s_{kp} = d_k\, l_{pk}$ formed once per panel: for $k$ ascending,
$l_{ik} = x_k / d_k$, then $x_p \leftarrow x_p - l_{ik}\, s_{kp}$ for all
$p > k$, so each entry subtracts its terms in ascending $k$, each term
rounded as $l_{ik} \cdot (d_k\, l_{pk})$. The update of each row $i \in R'$
depends only on that row, so rows are processed in independent strips; the
rows of $J'$ are solved first because $L_{J',P}$ is the shared operand of
the update.

### Work estimate used for engine selection

For a node of width $W$ and height $R$, counting one multiply-subtract per
updated entry over its $W$ pivots gives

$$\sum_{m=0}^{W-1} \left[ \frac{m(m+1)}{2} + (R - W)\, m \right]
  = \frac{W(W-1)(3R - 2W + 1)}{6}$$

terms (the first summand is the shrinking lower triangle of the diagonal
block, the second the rectangular tail). The blocked engine is selected only
when some node has $W \ge 320$ and the summed work of all such nodes is at
least $3072\, n$; narrower or lighter problems keep the simplicial kernel,
whose per-column overhead is lower.

## 7. Elimination tree and symbolic structure

For a symmetric pattern with Cholesky-like factor $L$, the elimination tree is

$$\mathrm{parent}(j) = \min\{\, i > j : l_{ij} \ne 0 \,\}.$$

The structure of column $j$ of $L$ satisfies
$\mathrm{Struct}(l_{:,j}) \setminus \{j\} \subseteq \mathrm{Struct}(l_{:,\mathrm{parent}(j)})$,
and row $k$ of $L$ is the row subtree described in Section 6. The code
computes `parent` and the strict column counts $c_j = |\mathrm{Struct}(l_{:,j})| - 1$
in one pass over the input: for each column $k$ and each $b_{ik} \ne 0$ with
$i < k$, walk from $i$ towards the root until a node already marked with $k$
is met, setting the parent of unparented nodes to $k$ and incrementing the
count of every node passed. This is Liu's $O(\mathrm{nnz}(L))$ algorithm; the
same traversal, repeated, fills the exact row indices when the blocked kernel
needs them.

Consecutive columns with $\mathrm{parent}(j) = j+1$ and $c_j = c_{j+1} + 1$
form a fundamental supernode: the parent relation gives
$\mathrm{Struct}(l_{:,j}) \setminus \{j+1\} \subseteq \mathrm{Struct}(l_{:,j+1})$
and the count equality makes the inclusion an equality.

For nonsymmetric LU the AMD estimates of $\mathrm{nnz}(L)$ (`AMD_LNZ`) and of
the largest column count (`AMD_DMAX`) on the pattern of $A + A^T$ are used as
an a-priori bound: the left-looking CSC engine is selected when
`AMD_DMAX` $\le 192$ and the predicted fill
$2\,(\texttt{AMD\_LNZ} + n) / \mathrm{nnz}(A) \le 20$.

## 8. Fill-reducing orderings

- `NATURAL`: $P_c = I$.
- `MMD_AT_PLUS_A`: the vendored SuiteSparse AMD applied to the symmetric
  pattern of $A + A^T$. AMD is an approximate minimum-degree ordering on the
  quotient graph: eliminating pivot $p$ at step $k$ merges the elements
  adjacent to $p$ into one new element with vertex set $L_p$, and the true
  external degree of each neighbour $i$ is replaced by the upper bound

  $$\bar d_i = \min\Big(\, n - k,\;
    \bar d_i^{\,\mathrm{old}} + |L_p \setminus \{i\}|,\;
    |A_i \setminus \{i\}| + |L_p \setminus \{i\}| + \sum_{e \in \mathcal{E}_i,\ e \ne p} |L_e \setminus L_p| \Big),$$

  where $A_i$ are the original neighbours of $i$ not yet absorbed and
  $\mathcal{E}_i$ the elements adjacent to $i$. The bound is exact for many
  graphs and is computed in time proportional to the sizes of the elements
  touched, which is what makes AMD fast. The option name follows SuperLU's
  `MMD_AT_PLUS_A`; the implementation is AMD, which also returns the
  $\mathrm{nnz}(L)$ and largest-front estimates used in Section 7.
- `COLAMD`: a column ordering for unsymmetric LU with row pivoting. For any
  row permutation $P_r$, $\mathrm{Struct}(L + U) \subseteq \mathrm{Struct}(R)$
  where $R$ is the Cholesky factor of $A^T A$ (George and Ng), so a good
  symmetric ordering of $A^T A$ bounds the fill of LU with partial pivoting.
  COLAMD computes an approximate minimum-degree ordering of the graph of
  $A^T A$ without forming the product, treating each row of $A$ as a clique on
  its column indices.

Both orderings are heuristics; the resulting factor sizes and the
`largest_front` statistic are what the engine selection acts on.

## 9. Triangular solver

`spsolve_triangular` works on CSR storage and performs row-oriented
substitution. For a lower triangular $A$ with right-hand side $b$:

$$x_i = \frac{1}{a_{ii}} \Big( b_i - \sum_{j < i} a_{ij}\, x_j \Big), \qquad i = 0, 1, \dots, n-1,$$

and for an upper triangular $A$ the same with $j > i$ and $i$ descending.
With `unit_diagonal=True` the division is skipped and stored diagonal values
are ignored. The structure is validated first (an entry with a nonzero value
on the wrong side of the diagonal is rejected, a zero or non-finite diagonal
is singular). A complex right-hand side with a real matrix is solved as two
real right-hand sides (real and imaginary parts), since the substitution is
linear with real coefficients. Independent right-hand sides are processed in
blocks of four (C order) or singly (Fortran order), in parallel.

## 10. Iterative refinement and error measures

After the initial solve $x_0$, up to two corrections are applied:

$$r_i = b - A x_i, \qquad x_{i+1} = x_i + (LU)^{-1} r_i,$$

using the stored factors (or $(LU)^{-T}$ for transposed solves) and the
original, unscaled matrix for the residual. Convergence is measured by the
componentwise backward error of Oettli and Prager,

$$\omega(x) = \max_i \frac{|b - Ax|_i}{(|A|\,|x| + |b|)_i},$$

the smallest $\epsilon$ such that $x$ solves $(A + \Delta A) x = b + \Delta b$
for some perturbation with $|\Delta A| \le \epsilon |A|$ and
$|\Delta b| \le \epsilon |b|$. A zero denominator certifies only an exact zero
residual. The iteration stops when $\omega \le 10\,\mathrm{eps}$
($\mathrm{eps} = 2^{-52}$); a correction that does not decrease $\omega$ or
makes it non-finite is discarded in favour of the previous iterate, and a
`RuntimeWarning` reports non-convergence. Skeel's analysis shows one step of
refinement in working precision brings $\omega$ to $O(\mathrm{eps})$ unless
the system is too ill-conditioned for that precision, which is why two steps
suffice in practice.

The benchmark reports the normwise variant

$$\frac{\|b - Ax\|_\infty}{\big\|\,|A|\,|x|\,\big\|_\infty + \|b\|_\infty} \le 10^{-12},$$

computed on the solve without refinement. Neither quantity is a forward
error: for an ill-conditioned matrix the solution can be far from the true
one while both are tiny. No condition estimate is computed.

## 11. Rounding, determinism and non-finite detection

- Every engine performs each output entry's accumulation in one fixed order
  (ascending pivot or contributor index), so results are bitwise reproducible
  across thread counts. Parallel work is always over disjoint output sets.
- The dense tile kernels (`src/dense_kernels.cpp`,
  `src/dense_kernels_avx2.cpp`) use one of two fixed accumulation orders.
  The blocked LU flush updates every output from its current value,
  $c \leftarrow (\cdots((c - a_0 b_0) - a_1 b_1) \cdots) - a_{k-1} b_{k-1}$,
  each term a rounded product and a rounded subtraction in every kernel: this
  is the recurrence of the panel block's rank-1 updates and of pivot-tail
  materialization, which apply the same terms through `axpy_sub` or, for
  sparse pivot rows during materialization, in scalar code, so a duplicated
  row still cancels exactly. The supernodal $LDL^T$ rank-$k$
  update ($k$ is a whole contributor's width, or at most 32 for an in-node
  panel update) forms $s = (\cdots((0 + a_0 b_0) + a_1 b_1) \cdots) + a_{k-1}
  b_{k-1}$ and then $c \leftarrow c - s$: when $|c|$ is much larger than the
  terms, as on a dominant diagonal, the running form would round at the scale
  of $c$ once per term instead of once in total. The `sse2` and `scalar`
  kernels round each product and each addition (two roundings per term); the
  `avx2-fma` kernel forms each $s \leftarrow \mathrm{fma}(a_k, b_k, s)$ with
  one rounding. In both orders an element's result depends only on its own
  operands, so tiling, edge handling, in-place versus gathered operands and
  the executing thread cannot change it. LU factors agree bitwise across the
  kernels on x86-64; $LDL^T$ factors from `avx2-fma` differ in the last bits,
  so kernels are selectable explicitly (`set_dense_kernel`,
  `SPSOLVE_FORCE_KERNEL`) and cross-machine reproducibility remains possible.
- Non-finite values cannot be produced by finite input without overflow or a
  zero pivot. The engines check the pivot column maximum and pivot, the
  multipliers, and every updated entry (or, where documented, every pivot
  candidate, every multiplier column and every retained row once; the
  row-hybrid panel path gives the completeness argument in
  `docs/native-lu.md`), and abort with a non-finite
  error, or a singular error if an exactly singular column is reached before
  a deferred check, before any factor is published. The supernodal $LDL^T$
  engine tests each pivot and scans each panel's L entries once: every node
  entry ends as a pivot or an L entry, the arithmetic keeps Inf and NaN
  non-finite, and each auxiliary product reaches an entry or a kernel flag
  (argument in `docs/native-lu.md`, "Blocked SPD panels"); a failed test
  falls back to pivoted LU. Pivot growth is bounded
  only by the usual $2^{n-1}$ worst case of partial pivoting; refinement, not
  pivoting, is the second line of defence against moderate growth.

## 12. Complexity summary

| Phase | Cost |
|---|---|
| Canonicalization, validation, equilibration, symmetry test | $O(\mathrm{nnz}(A) + n)$ |
| Row-hybrid setup (count, then fill) | $O(\mathrm{nnz}(A) + n)$ plus the initial dense tails |
| AMD / COLAMD ordering | near-linear in $\mathrm{nnz}(A)$ for typical graphs |
| Elimination tree and column counts | $O(\mathrm{nnz}(L))$ |
| Left-looking CSC LU | $O(\mathrm{flops}(LU) + \mathrm{nnz}(A))$ |
| Row-hybrid LU | $O(\mathrm{flops}(LU))$ for the updates plus $O(\mathrm{nnz}(\text{rows}))$ per sparse merge |
| Supernodal LDL^T | $O(\sum_{\text{nodes}} W(W-1)(3R-2W+1)/6)$ multiply-subtracts, dense-kernel rate |
| Solve | $O(\mathrm{nnz}(L + U) \cdot \text{nrhs})$ |
| Refinement | one sparse product and one solve per step |

## References

- J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time proportional
  to arithmetic operations," SIAM J. Sci. Stat. Comput. 9 (1988).
- J. W. H. Liu, "The role of elimination trees in sparse factorization,"
  SIAM J. Matrix Anal. Appl. 11 (1990).
- E. G. Ng and B. W. Peyton, "Block sparse Cholesky algorithms on advanced
  uniprocessor computers," SIAM J. Sci. Comput. 14 (1993).
- P. R. Amestoy, T. A. Davis and I. S. Duff, "An approximate minimum degree
  ordering algorithm," SIAM J. Matrix Anal. Appl. 17 (1996); and
  "Algorithm 837: AMD," ACM TOMS 30 (2004).
- T. A. Davis, J. R. Gilbert, S. I. Larimore and E. G. Ng, "A column
  approximate minimum degree ordering algorithm," ACM TOMS 30 (2004).
- A. George and E. Ng, "An implementation of Gaussian elimination with partial
  pivoting for sparse systems," SIAM J. Sci. Stat. Comput. 6 (1985).
- W. Oettli and W. Prager, "Compatibility of approximate solution of linear
  equations with given error bounds for coefficients and right-hand sides,"
  Numer. Math. 6 (1964).
- R. D. Skeel, "Iterative refinement implies numerical stability for Gaussian
  elimination," Math. Comp. 35 (1980).
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM (2002), chapters 9 and 12.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
  Hopkins (2013), section 3.2 (unblocked and blocked LU; the panel block
  follows the unblocked right-looking form, as in LAPACK's `dgetf2`).
- T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM (2006),
  chapters 4 to 7.
