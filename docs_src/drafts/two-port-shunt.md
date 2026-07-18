# State-space representation of a shunt acoustic impedance (e.g., yielding wall, thermal loss, air compression)

A shunt (per-unit-length) acoustic impedance modeled by an LTI transfer function $Z_w(s) = P(s)/U_w(s)$. This model may represent yielding wall, thermal loss, air compression, or a combination thereof. Given the acoustic pressure of $P$ in this segment, some of its flow "leaks" to the tube wall at a rate $U_w$, resulting in a difference between the input flow $U_1$ and output flow $U_2$:

$$
U_1 = U_2 + U_w
$$

The partial pressures are governed by

$$
\begin{align}
F_1 + B_1 &= F_2 + B_2\\
\frac{F_1-B_1}{Z} &= \frac{F_2-B_2}{Z} +U_w\\
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

## Case 1: Proper $Y_w(s)$

Instead of the impedance $Z_w$, it is often more algorithmically efficient to use 
the admittance $Y_w = 1/Z_w$ when a tube segment model combines multiple impedance 
elements in parallel (parallel in impedance = series in admittance).

First, if $Y_w(s)$ is a proper transfer function, it has a state-space representation of $Y_w$, denoted by

$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A}_y \mathbf{s} + \mathbf{b}_y P\\
U_w &= \mathbf{c}_y \mathbf{s} + d_y P\\
\end{align}
$$

and the three governing equations are expressed with the partial pressures and the total pressure $P$:

$$
\begin{align}
  F_1+B_1 &= F_2+B_2\\
  F_2 + B_2 &= P\\
  Z^{-1}F_1 - Z^{-1}B_1 &= Z^{-1}F_2 - Z^{-1}B_2 + \mathbf{c}_y \mathbf{s} + d_y P\\
\end{align}
$$

By gathering the outputs ($F_2$, $B_1$, and $P$) to the left hand and the inputs ($F_1$, $B_2$, $\mathbf{s}$) to the right hand, we get the system equations in a vector-matrix format:

$$
\begin{bmatrix}
  1 & -1 & 0\\
  1 & 0 & -1\\
  Z^{-1} & Z^{-1} & d_y\\
\end{bmatrix}
\begin{bmatrix}
  F_2\\B_1\\P
\end{bmatrix}=
\begin{bmatrix}
  \mathbf{0}\\\mathbf{0}\\-\mathbf{c}_y
\end{bmatrix}
\mathbf{s}
+\begin{bmatrix}
  1 & -1\\
  0 & -1\\
  Z^{-1} & Z^{-1}\\
\end{bmatrix}
\begin{bmatrix}
  F_1\\B_2
\end{bmatrix}
$$

Solving this equation yields the output equation matrices and $P=\mathbf{c}_v\mathbf{s}+\mathbf{d}_v[F_1\ B_2]^T$:

$$
\begin{align}
\begin{bmatrix}
  \mathbf{C}\\\mathbf{c}_v
\end{bmatrix}&=
\begin{bmatrix}
  1 & -1 & 0\\
  1 & 0 & -1\\
  Z^{-1} & Z^{-1} & d_y\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  \mathbf{0}\\\mathbf{0}\\-\mathbf{c}_y
\end{bmatrix}\\
\begin{bmatrix}
  \mathbf{D}\\\mathbf{d}_v
\end{bmatrix}&=
\begin{bmatrix}
  1 & -1 & 0\\
  1 & 0 & -1\\
  Z^{-1} & Z^{-1} & d_y\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  1 & -1\\
  0 & -1\\
  Z^{-1} & Z^{-1}\\
\end{bmatrix}
\end{align}
$$

and the two-port state equation matrices:

$$
\begin{align}
\mathbf{A} &= \mathbf{A}_y + \mathbf{b}_y\mathbf{c}_v\\
\mathbf{B} &= \mathbf{b}_y\mathbf{d}_v
\end{align}
$$

## Case 2: Improper $Y_w(s)$

If $Y_w(s)$ is proper, the solution must be acquired with the impedance $Z_w(s) = 1/Y_w(s)$. Let a state-space representation of $Z_w(s)$ to be

$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A}_w \mathbf{s} + \mathbf{b}_w U_w\\
P &= \mathbf{c}_w \mathbf{s} + d_w U_w\\
\end{align}
$$

Then the governing exuations are

$$
\begin{align}
  F_1+B_1 &= F_2+B_2\\
  F_2 + B_2 &= \mathbf{c}_w \mathbf{s} + d_w U_w\\
  Z^{-1}F_1 - Z^{-1}B_1 &= Z^{-1}F_2 - Z^{-1}B_2 + U_w\\
\end{align}
$$

Substituting (4) into (1), we get a system of  3 equations:

$$
\begin{bmatrix}
  1 & -1 & 0\\
  1 & 0 & -d_w\\
  Z^{-1} & Z^{-1} & 1\\
\end{bmatrix}
\begin{bmatrix}
  F_2\\B_1\\U_w
\end{bmatrix}=
\begin{bmatrix}
  \mathbf{0}\\\mathbf{c}_w\\\mathbf{0}
\end{bmatrix}
\mathbf{s}
+\begin{bmatrix}
  1 & -1\\
  0 & -1\\
  Z^{-1} & Z^{-1}\\
\end{bmatrix}
\begin{bmatrix}
  F_1\\B_2
\end{bmatrix}
$$

Solving this yields

$$
\begin{equation}
\begin{bmatrix}
  F_2\\B_1\\U_w
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
  1 & -1 & 0\\
  1 & 0 & -d_w\\
  Z^{-1} & Z^{-1} & 1\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  \mathbf{0}\\\mathbf{c}_w\\\mathbf{0}
\end{bmatrix}\\
\begin{bmatrix}
  \mathbf{D}\\\mathbf{d}_u
\end{bmatrix}&=
\begin{bmatrix}
  1 & -1 & 0\\
  1 & 0 & -d_w\\
  Z^{-1} & Z^{-1} & 1\\
\end{bmatrix}^{-1}
\begin{bmatrix}
  1 & -1\\
  0 & -1\\
  Z^{-1} & Z^{-1}\\
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
