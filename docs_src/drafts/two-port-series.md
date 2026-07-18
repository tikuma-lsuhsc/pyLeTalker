# State-space representation of a series acoustic impedance (propagation delay/viscous/laminar loss)

A series (per-unit-length) acoustic impedance modeled by an LTI transfer function $Z_p(s) = P_p(s)/U(s)$. This model may represent signal propagation, viscous loss, a laminar loss, or a combination thereof. As the acoustic wave flows through a segment at a rate $U$, its input pressure $P_1$ and the output pressure $P_2$ differ and satisfies the pressure conseration:

$$
P_1 = P_p + P_2 = Z_pU + P_2
$$

The partial pressures are governed by

$$
\begin{align}
F_1 + B_1 &= P_p + F_2 + B_2\\
\frac{F_1-B_1}{Z} &= \frac{F_2-B_2}{Z} = U\\
\end{align}
$$

where $Z = \rho c / A$ with the tube segment's cross-sectional area $A$.

We want to find a two-port representation of  this series impedance in the form:

$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A} \mathbf{s} + \mathbf{B}\begin{bmatrix}F_1\\B_2\end{bmatrix}\\
\begin{bmatrix}F_2\\B_1\end{bmatrix} &= \mathbf{C} \mathbf{s} + \mathbf{D} \begin{bmatrix}F_1\\B_2\end{bmatrix}\\
\end{align}
$$

## Case 1: Proper $Z_p(s)$

If $Z_p(s)$ is proper, there exists a state-space representation:

$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A}_p \mathbf{s} + \mathbf{b}_p U\\
P_p &= \mathbf{c}_p \mathbf{s} + d_p U\\
\end{align}
$$

With these equation, we now have a system of three equations to solve:

$$
\begin{align}
F_1 + B_1 &= F_2 + B_2 + \mathbf{c}_p \mathbf{s} + d_p U\\
\frac{F_1-B_1}{Z} &= \frac{F_2-B_2}{Z} \\
\frac{F_2-B_2}{Z} &= U\\
\end{align}
$$

In matrix-vector formulation, we get

$$
\begin{bmatrix}
  1 & -1 & d_p\\
  Z^{-1} & Z^{-1} & 0\\
  Z^{-1} & 0 & -1\\
\end{bmatrix}
\begin{bmatrix}
  F_2\\B_1\\U
\end{bmatrix}=
\begin{bmatrix}
  -\mathbf{c}_p\\\mathbf{0}\\\mathbf{0}
\end{bmatrix}
\mathbf{s}
+\begin{bmatrix}
  1 & -1\\
  Z^{-1} & Z^{-1}\\
  0 & Z^{-1}
\end{bmatrix}
\begin{bmatrix}
  F_1\\B_2
\end{bmatrix}
$$

Solving this yields

$$
\begin{equation}
\begin{bmatrix}
  F_2\\B_1\\U
\end{bmatrix}=
\begin{bmatrix}
  \mathbf{C}\\
  \mathbf{c}_u\\
\end{bmatrix}
\mathbf{s}
+\begin{bmatrix}
  \mathbf{D}\\
  \mathbf{d}_u\\
\end{bmatrix}
\begin{bmatrix}
  F_1\\B_2
\end{bmatrix}
\end{equation}
$$

where

$$
\begin{align}
\begin{bmatrix}
  \mathbf{C}\\\mathbf{c}_u
\end{bmatrix}&=
\begin{bmatrix}
  1 & -1 & d_p\\
  Z^{-1} & Z^{-1} & 0\\
  Z^{-1} & 0 & -1\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  -\mathbf{c}_p\\\mathbf{0}\\\mathbf{0}
\end{bmatrix}\\
\begin{bmatrix}
  \mathbf{D}\\\mathbf{d}_u
\end{bmatrix}&=
\begin{bmatrix}
  1 & -1 & d_p\\
  0 & Z^{-1} & 1\\
  Z^{-1} & 0 & -1\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  1 & -1\\
  Z^{-1} & Z^{-1}\\
  0 & Z^{-1}
\end{bmatrix}
\end{align}
$$

The two-port state equation is found by substituting the $U=\mathbf{c}_u\mathbf{s}+\mathbf{d}_u[F_1\ B_2]^T$ into (5) with the matrices:

$$
\begin{align}
\mathbf{A} &= \mathbf{A}_p + \mathbf{b}_p\mathbf{c}_u\\
\mathbf{B} &= \mathbf{b}_p\mathbf{d}_u
\end{align}
$$

## Case 2: Improper $Z_p(s)$

If $Z_p(s)$ is improper, a two-port state-space representation must be sought with the admittance $Y_p = 1/Z_p$. (This is the more common case as the series element is usually inductive). Let a state-space representation of $Y_p$ be

$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A}_y \mathbf{s} + \mathbf{b}_y P_p\\
U &= \mathbf{c}_y \mathbf{s} + d_y P_p\\
\end{align}
$$

and the three governing equations are:

$$
\begin{align}
  F_1+B_1 &= F_2+B_2+P_p\\
  Z^{-1}F_1 - Z^{-1}B_1 &= Z^{-1}F_2 - Z^{-1}B_2\\
  Z^{-1}F_2 - Z^{-1}B_2 &= \mathbf{c}_y \mathbf{s} + d_y P_p\\
\end{align}
$$

Formulating them as a vector-matrix format:

$$
\begin{bmatrix}
  1 & -1 & 1\\
  Z^{-1} & Z^{-1} & 0\\
  Z^{-1} & 0 & -d_y\\
\end{bmatrix}
\begin{bmatrix}
  F_2\\B_1\\P_p
\end{bmatrix}=
\begin{bmatrix}
  \mathbf{0}\\\mathbf{0}\\\mathbf{c}_y
\end{bmatrix}
\mathbf{s}
+\begin{bmatrix}
  1 & -1\\
  Z^{-1} & Z^{-1}\\
  0 & Z^{-1}
\end{bmatrix}
\begin{bmatrix}
  F_1\\B_2
\end{bmatrix}
$$

Solving this equation yields the output equation matrices and $P_p=\mathbf{c}_v\mathbf{s}+\mathbf{d}_v[F_1\ B_2]^T$:

$$
\begin{align}
\begin{bmatrix}
  \mathbf{C}\\\mathbf{c}_v
\end{bmatrix}&=
\begin{bmatrix}
  1 & -1 & 1\\
  Z^{-1} & Z^{-1} & 0\\
  Z^{-1} & 0 & -d_y\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  \mathbf{0}\\\mathbf{0}\\\mathbf{c}_y
\end{bmatrix}\\
\begin{bmatrix}
  \mathbf{D}\\\mathbf{d}_v
\end{bmatrix}&=
\begin{bmatrix}
  1 & -1 & 1\\
  Z^{-1} & Z^{-1} & 0\\
  Z^{-1} & 0 & -d_y\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  1 & -1\\
  Z^{-1} & Z^{-1}\\
  0 & Z^{-1}
\end{bmatrix}
\end{align}
$$

and the two-port state equation matrices:

$$
\begin{align}
\mathbf{A} &= \mathbf{A}_p + \mathbf{b}_y\mathbf{c}_v\\
\mathbf{B} &= \mathbf{b}_p\mathbf{d}_v
\end{align}
$$
