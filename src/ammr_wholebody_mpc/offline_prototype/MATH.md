# SE(2) GMPC — Mathematical Reference

> **Purpose.** Every formula used by the offline GMPC prototype, with its
> derivation, primary literature citation, and the exact code location that
> implements it. Use this when writing the thesis method chapter or when
> debugging numerical mismatches.

> **Honest scope warning.** All formulas below are *standard* in robotics
> Lie-group / linear-MPC literature. They were transcribed from memory and
> validated by `_selftest` in each module — but **self-tests catch obvious
> sign/scale errors, not subtle convention mismatches**. Before citing in
> the thesis, verify against the listed primary reference.

---

## 0. Notation and conventions

| Symbol            | Meaning                                                      |
| ----------------- | ------------------------------------------------------------ |
| $X \in SE(2)$     | Chassis pose (3×3 homogeneous matrix)                        |
| $\xi = (v_x, v_y, \omega) \in \mathfrak{se}(2)$ | **Body-frame** body twist          |
| $\hat{\xi}$       | Algebra element corresponding to $\xi$ (3×3 matrix)          |
| $R(\theta)$       | 2×2 SO(2) rotation matrix                                    |
| $J = \begin{bmatrix}0 & -1\\ 1 & 0\end{bmatrix}$ | SO(2) generator                |
| $\mathrm{Ad}_X$   | Big Adjoint, 3×3 matrix on $\mathfrak{se}(2)$                |
| $\mathrm{ad}(\xi)$| Lie-bracket as a 3×3 matrix on $\mathfrak{se}(2)$            |
| $e$               | Body-frame geodesic error                                    |
| $\delta\xi_k$     | Decision variable: deviation from reference twist            |

**Convention (body twist):** kinematics are
$$\dot X = X \cdot \hat\xi \qquad (1)$$
This matches an Omni chassis whose `/cmd_vel` is expressed in the body frame.
Lynch & Park §3.3.2 calls this the *body twist convention*; Solà 18 §10.3.1
denotes it $X \xi^\wedge$.

---

## 1. SE(2) hat operator

$$\hat{\xi} \;=\; \begin{bmatrix} 0 & -\omega & v_x \\ \omega & 0 & v_y \\ 0 & 0 & 0 \end{bmatrix}$$

**Derivation.** $\mathfrak{se}(2)$ is spanned by three generators
$\{G_1, G_2, G_3\}$ where $G_1, G_2$ are translations along body $x, y$
and $G_3$ is the rotation generator. The formula above is just
$\xi^\wedge = v_x G_1 + v_y G_2 + \omega G_3$.

- **Reference:** Solà 18, eq. (140); Lynch & Park §3.3.1.
- **Code:** [se2.py:54 `hat()`](se2.py#L54), [se2.py:67 `vee()`](se2.py#L67).

---

## 2. SE(2) exponential map  $\exp: \mathfrak{se}(2) \to SE(2)$

For $\xi = (v, \omega)$ with $v = (v_x, v_y)$:

$$\exp(\hat\xi) \;=\; \begin{bmatrix} R(\omega) & V(\omega) v \\ 0 & 1 \end{bmatrix}, \qquad V(\omega) \;=\; \frac{1}{\omega}\begin{bmatrix} \sin\omega & -(1-\cos\omega) \\ 1-\cos\omega & \sin\omega \end{bmatrix}$$

**Derivation.** Integrate $\dot X = X \hat\xi$ from $X(0) = I$ over one
unit of time with constant $\xi$. The rotation block is the standard 2-D
rotation. The translation block accumulates body-frame velocity rotated by
the time-varying body orientation, giving the integral
$\int_0^1 R(s\omega) \, v\, ds = V(\omega)v$.

**Small-$\omega$ Taylor:** $V(\omega) = I + \tfrac{\omega}{2} J + O(\omega^2)$
— required to avoid 0/0 when $\omega \to 0$.

- **Reference:** Solà 18, eqs. (143)–(145).
- **Code:** [se2.py:95 `exp_()`](se2.py#L95); helper $V$ in
  [se2.py:79 `_V()`](se2.py#L79) handles the Taylor branch when
  $|\omega| < 10^{-8}$.

---

## 3. SE(2) logarithm  $\log : SE(2) \to \mathfrak{se}(2)$

For $X = (R(\theta), p)$:

$$\theta = \operatorname{atan2}(R_{21}, R_{11}), \qquad v = V(\theta)^{-1} p$$

with closed form

$$V(\theta)^{-1} \;=\; \frac{1}{2}\begin{bmatrix} \theta \cot(\theta/2) & \theta \\ -\theta & \theta\cot(\theta/2) \end{bmatrix}$$

- **Reference:** Solà 18, eqs. (146)–(149).
- **Code:** [se2.py:107 `log_()`](se2.py#L107).

**Caveat (branch cut).** $\log$ returns $\theta \in (-\pi, \pi]$. So
`exp(log(X)) = X` *as group elements* (always), but the vector identity
`log(exp(ξ)) = ξ` only holds when $\omega \in (-\pi, \pi]$. This is by
design (the log map gives the *shortest* geodesic) and is exactly what
defeats the wrap-around bug of the prior $(x, y, \theta)$-state MPC.

---

## 4. Big Adjoint  $\mathrm{Ad}_X$

$$\mathrm{Ad}_X \;=\; \begin{bmatrix} R & -J p \\ 0 & 1 \end{bmatrix}, \qquad J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

**Definition.** $\mathrm{Ad}_X$ is the unique linear map such that
$X \exp(\hat\xi) X^{-1} = \exp\big((\mathrm{Ad}_X \xi)^\wedge\big)$.

**Sign warning.** The $-Jp$ block is correct; my first draft had $+Jp$ and
failed the conjugation self-test (#5 in [se2.py:_selftest](se2.py#L213)).
The sign was fixed by direct algebraic verification against
$X \cdot \exp((0,0,1)^\wedge) \cdot X^{-1}$ for $X = \mathrm{Trans}(1, 0)$.

- **Reference:** Lynch & Park §3.3.2 (SE(3) version); reduced to SE(2)
  by restricting to planar motion. Solà 18, eq. (159).
- **Code:** [se2.py:149 `Ad()`](se2.py#L149).

---

## 5. Lowercase adjoint  $\mathrm{ad}(\xi)$

$$\mathrm{ad}(\xi) \;=\; \begin{bmatrix} 0 & -\omega & v_y \\ \omega & 0 & -v_x \\ 0 & 0 & 0 \end{bmatrix}$$

**Defining identity.** $\mathrm{ad}(\xi_1)\,\xi_2 = [\xi_1, \xi_2]^\vee$,
i.e. matches the Lie bracket on $\mathfrak{se}(2)$.

**Derivation.** Computed by hand from
$\hat\xi_1 \hat\xi_2 - \hat\xi_2 \hat\xi_1$ entry by entry, then expressed
as a 3×3 matrix on the basis $(v_x, v_y, \omega)$. The result is
verified numerically against the Lie bracket in
[se2.py:_selftest](se2.py#L213) test #4 (20 random twist pairs).

- **Reference:** Solà 18, eq. (175); Lynch & Park §8.2.2.
- **Code:** [se2.py:170 `ad()`](se2.py#L170).

---

## 6. Geodesic body-frame error

$$e \;=\; \log\!\Big( X_{\mathrm{ref}}^{-1} \cdot X \Big)^{\!\vee}\;\in\;\mathfrak{se}(2)$$

**Properties** (used by GMPC):

1. $e = 0 \iff X = X_{\mathrm{ref}}$
2. $e$ never wraps in $\omega$ — see §3.
3. Linearisation gives clean LTI-with-LTV-A error dynamics — see §7.

- **Code:** [se2.py:189 `geodesic_error()`](se2.py#L189).

---

## 7. Continuous error dynamics

**Claim.**
$$\dot e \;=\; -\,\mathrm{ad}(\xi_{\mathrm{ref}})\, e \;+\; (\xi - \xi_{\mathrm{ref}}) \qquad (2)$$

**Derivation.** Both trajectories obey (1), so

$$
\dot X = X\hat\xi,\qquad \dot X_{\mathrm{ref}} = X_{\mathrm{ref}} \hat\xi_{\mathrm{ref}}
$$

Differentiating $X_{\mathrm{ref}}^{-1}$ via $\partial_t X^{-1} = -X^{-1} \dot X X^{-1}$:

$$
\partial_t X_{\mathrm{ref}}^{-1} \;=\; -\hat\xi_{\mathrm{ref}} X_{\mathrm{ref}}^{-1}
$$

so with $E = X_{\mathrm{ref}}^{-1} X$:

$$
\dot E \;=\; -\hat\xi_{\mathrm{ref}}\,E \;+\; E\,\hat\xi \qquad (3)
$$

Write $E \approx I + \hat e$ for small error. Substitute and drop products
of small quantities:

$$
\dot{\hat e} \;\approx\; (\hat\xi - \hat\xi_{\mathrm{ref}}) \;+\; \hat e \, \hat\xi_{\mathrm{ref}} - \hat\xi_{\mathrm{ref}} \hat e
\;=\; \widehat{(\xi - \xi_{\mathrm{ref}})} \;-\; \widehat{\mathrm{ad}(\xi_{\mathrm{ref}})\,e}
$$

Applying $\vee$ gives (2). $\square$

**Citation status (HONEST).** This is the standard *Lie-group error
dynamics* pattern used in:

- Bullo & Lewis (2005) *Geometric Control of Mechanical Systems*
- Mahony, Hamel, Pflimlin (2008) *"Nonlinear Complementary Filters on the
  Special Orthogonal Group"* — SO(3) analogue
- Teng et al. (2021) *"Lie-Theoretic Kalman Filtering for Inertial Integrated
  Navigation"* — SE(3) analogue
- Phogat, Chatterjee, Banavar (2018) *"Discrete-Time Optimal Attitude Control of a Spacecraft with Momentum and Control Constraints"* — closer to MPC application

For the thesis, **cite Bullo & Lewis or Mahony 08 for the underlying
identity**, then **derive (2) explicitly as above** so the reader doesn't
need to chase a SO(3)/SE(3) → SE(2) reduction.

---

## 8. Discrete-time error model

Forward Euler discretisation of (2) with step $\Delta t$:

$$e_{k+1} \;=\; A_d(k)\,e_k \;+\; \Delta t\,\delta\xi_k$$
$$A_d(k) \;=\; I \;-\; \Delta t \cdot \mathrm{ad}\!\big(\xi_{\mathrm{ref}}(k)\big)$$
$$\delta\xi_k \;\triangleq\; \xi_k - \xi_{\mathrm{ref}}(k)$$

**Note.** $A_d$ depends on $k$ through $\xi_{\mathrm{ref}}(k)$ — this is
the **multi-point linearisation along the reference**. This is the direct
fix for the failure mode of the previous $(x, y, \theta)$-state MPC, which
linearised only around the *current* state.

- **Reference:** Forward Euler — undergraduate numerical methods. No
  paper citation needed; just label "Forward Euler" in the thesis.
- **Code:** see `A_d` construction in
  [gmpc.py:198 `GMPC.solve()`](gmpc.py#L198).

---

## 9. Prediction matrices (condensed MPC form)

Stack $E = [e_1; e_2; \ldots; e_N]$ and $z = [\delta\xi_0; \ldots; \delta\xi_{N-1}]$:

$$E \;=\; \Phi\,e_0 \;+\; \Gamma\,z$$

where

$$
\Phi[i,:] = \prod_{k=0}^{i} A_d(k), \qquad
\Gamma[i,j] = \begin{cases} \big(\prod_{k=j+1}^{i} A_d(k)\big)\,\Delta t \cdot I & j \le i \\ 0 & j > i \end{cases}
$$

Built iteratively via the prefix-product recurrence:

$$
\Phi[i] = A_d(i)\,\Phi[i-1], \qquad
\Gamma[i,j<i] = A_d(i)\,\Gamma[i-1,j], \qquad
\Gamma[i,i] = \Delta t \cdot I.
$$

- **Reference:** Borrelli, Bemporad, Morari (2017) *Predictive Control for
  Linear and Hybrid Systems*, §11.3 (condensed form).
- **Code:** [gmpc.py:89 `_build_prediction()`](gmpc.py#L89).

---

## 10. Cost function

$$
J \;=\; \sum_{k=1}^{N-1} e_k^\top Q\, e_k \;+\; e_N^\top Q_f\, e_N \;+\; \sum_{k=0}^{N-1} \delta\xi_k^\top R\, \delta\xi_k
$$

With $\bar Q = \mathrm{blkdiag}(Q,\ldots,Q, Q_f)$ and $\bar R = \mathrm{blkdiag}(R,\ldots,R)$, substituting $E = \Phi e_0 + \Gamma z$:

$$
J \;=\; z^\top \big(\Gamma^\top \bar Q \Gamma + \bar R\big) z \;+\; 2\,z^\top \Gamma^\top \bar Q \Phi\, e_0 \;+\; \text{const}
$$

Matched to OSQP's standard form $\frac{1}{2} z^\top P z + q^\top z$:

$$
P \;=\; 2\,(\Gamma^\top \bar Q \Gamma + \bar R), \qquad
q \;=\; 2\,\Gamma^\top \bar Q \Phi\, e_0
$$

**Factor-of-2 origin.** Comes from the $\frac{1}{2}$ in OSQP's standard
form; some MPC texts omit it and write $z^\top P z + q^\top z$ instead.
If you cite Borrelli 17 directly verbatim, check which convention they use.

- **Reference:** Borrelli 17 §11.3 (cost condensation); OSQP docs (standard
  form).
- **Code:** [gmpc.py:114 `_build_Q_bar()`](gmpc.py#L114),
  [gmpc.py:124 `_build_R_bar()`](gmpc.py#L124),
  $P$ and $q$ construction in
  [gmpc.py:223 `GMPC.solve()` step 4](gmpc.py#L223).

---

## 11. Constraints

### 11a. Velocity box

$$u_{\min} \;\le\; \xi_{\mathrm{ref}}(k) + \delta\xi_k \;\le\; u_{\max}, \qquad k = 0, \ldots, N-1$$

Rewritten in $z$ coordinates:

$$u_{\min} - \xi_{\mathrm{ref}}(k) \;\le\; \delta\xi_k \;\le\; u_{\max} - \xi_{\mathrm{ref}}(k)$$

Implemented as the identity block in $A_{\mathrm{total}}$ with bounds shifted
by $\xi_{\mathrm{ref}}(k)$.

### 11b. Acceleration box (finite-difference)

$$|u_k - u_{k-1}| \;\le\; a_{\max}\, \Delta t, \qquad u_{-1} \;\triangleq\; \xi_{\mathrm{prev}}$$

In $z$ coordinates, for $k \ge 1$:

$$
-a_{\max}\Delta t - \xi_{\mathrm{ref}}(k) + \xi_{\mathrm{ref}}(k{-}1)
\;\le\; \delta\xi_k - \delta\xi_{k-1}
\;\le\; a_{\max}\Delta t - \xi_{\mathrm{ref}}(k) + \xi_{\mathrm{ref}}(k{-}1)
$$

For $k=0$ replace $\xi_{\mathrm{ref}}(k-1)$ with $\xi_{\mathrm{prev}}$ and
$\delta\xi_{k-1}$ with $0$.

Stacked into OSQP form $l \le A_{\mathrm{total}} z \le u$.

- **Reference:** No specific paper — this is just box + finite-difference
  constraints, standard MPC practice. Cite Borrelli 17 §10 (or any MPC
  textbook) for completeness.
- **Code:** [gmpc.py:137 `_build_constraints()`](gmpc.py#L137).

---

## 12. Suggested thesis citation set (minimum)

Drop these into the bibliography and cite at the listed sections:

| Cite                    | For sections | Reason                                             |
| ----------------------- | ------------ | -------------------------------------------------- |
| Solà, Deray, Atchuthan (2018) "A micro Lie theory…" | 1–6 | Modern standard SE(2)/SE(3) Lie reference         |
| Lynch & Park (2017) *Modern Robotics*    | 0, 1, 4, 5 | Body-twist convention; SE(2) Adjoint               |
| Bullo & Lewis (2005) *Geometric Control…* | 7        | Lie-group error dynamics pattern                   |
| Mahony, Hamel, Pflimlin (2008) IEEE TAC  | 7         | SO(3) analogue of (2) — supporting evidence       |
| Borrelli, Bemporad, Morari (2017) *Predictive Control for Linear and Hybrid Systems* | 8–11 | Condensed-form linear MPC; constraint handling     |
| Stellato et al. (2020) "OSQP: An Operator Splitting Solver for Quadratic Programs" *Mathematical Programming Computation* | 11, results | Solver citation for the reported solve times       |

**What's still missing.** A paper that does **exactly SE(2)-error MPC for a
holonomic mobile base**. If one exists you should cite it as prior art. If
not, *this is your contribution* — write it as such.

---

## 13. Self-test coverage

What `_selftest` proves vs. what it does **not** prove:

| Property                                              | Proved by `_selftest`? |
| ----------------------------------------------------- | ---------------------- |
| `vee ∘ hat = id`                                      | ✓ test 1               |
| `log ∘ exp = id` (on principal branch)                | ✓ test 2a              |
| `exp ∘ log = id` (group)                              | ✓ test 2b              |
| `X · X⁻¹ = I`                                         | ✓ test 3               |
| `ad(ξ₁)ξ₂ = [ξ₁, ξ₂]^∨`                               | ✓ test 4               |
| `X · exp(ξ̂) · X⁻¹ = exp((Ad·ξ)^)` (Adjoint identity) | ✓ test 5               |
| `geodesic_error(X, X) = 0`                            | ✓ test 6a              |
| `geodesic_error` rotates into body frame correctly    | ✓ test 6c              |
| Error dynamics (2) is correct                         | **✗** (not tested — implicit via successful closed-loop tracking in `run.py`) |
| Prediction matrix $\Gamma$ is built correctly         | **✗** (not unit-tested — verified only end-to-end) |
| Constraint signs are correct                          | **✗** (not unit-tested — verified only end-to-end) |

**TODO (before publishing):** Add direct unit tests for the three rows
marked ✗, so a reviewer / future-you can trust correctness without
running the full closed-loop pipeline.

---

*Last updated when offline GMPC prototype reached DoD (5 trajectories, 0
infeasibility, RMSE bounded, solve time < 0.5 ms p95). Update before
adding CBF in Phase 4.*
