# State-space representation of a series acoustic impedance (propagation delay/viscous/laminar loss)

A series (per-unit-length) acoustic impedance modeled by an LTI transfer function $Z_p(s) = P_p(s)/U(s)$. This model may represent signal propagation, viscous loss, a laminar loss, or a combination thereof. As the acoustic wave flows through a segment at a rate $U$, its input pressure $P_1$ and the output pressure $P_2$ differ and satisfies the pressure conseration:
$$
P_1 = P_p + P_2 = Z_pU + P_2
$$
In terms of the partial pressures, we get
$$
\begin{align}
F_1 + B_1 &= Z_p U + F_2 + B_2\\
\frac{F_1-B_1}{Z} &= \frac{F_2-B_2}{Z} = U\\
\end{align}
$$
where $Z = \rho c / A$ with the tube segment's cross-sectional area $A$.





Let $\mathbf{A}_v$, $\mathbf{b}_v$, $\mathbf{c}_v$, and $d_v$
as the state-space matrices of this transfer function. (Though $H_v$ is a 
first-order system, we assume $H_v$ is of an arbitrary order.) Then, we have
$$\begin{align}
\dot{\mathbf{s}} &= \mathbf{A}_v \mathbf{s} + \mathbf{b}_v U\\
P_v &= \mathbf{c}_v \mathbf{s} + d_v U\\
\end{align}$$
We want to find an encompassing system:
$$\begin{align}
\dot{\mathbf{s}} &= \mathbf{A} \mathbf{s} + \mathbf{B}\begin{bmatrix}F_1\\B_2\end{bmatrix}\\
\begin{bmatrix}F_2\\B_1\end{bmatrix} &= \mathbf{C} \mathbf{s} + \mathbf{D} \begin{bmatrix}F_1\\B_2\end{bmatrix}\\
\end{align}$$

The governing equations are

$$\begin{align}
F_1+B_1 &= P_v + F_2 + B_2\\
\frac{1}{Z}(F_1-B_1) &= \frac{1}{Z}(F_2-B_2)
\end{align}$$

Here, the tube cross-sectional area is fixed so the impedance is a constant $Z$. Substitute (51) into (54) and express (54) and (55) for $F_2$ and $B_1$:

$$\begin{align}
\mathbf{c}_v \mathbf{s} + d_vZ^{-1} (F_2-B_2) + F_2 + B_2 &= F_1+B_1\\
(d_vZ^{-1}+1) F_2 - B_1 &= F_1 + (d_vZ^{-1}-1) B_2 - \mathbf{c}_v \mathbf{s}\\
\frac{1}{Z}(F_2 + B_1) &= \frac{1}{Z}(F_1+B_2)
\end{align}$$

Solve for $F_2$ and $B_1$ in a matrix-vector format:

$$\begin{aligned}
\begin{bmatrix}
d_vZ^{-1}+1 & -1\\
Z^{-1} & Z^{-1}
\end{bmatrix}
\begin{bmatrix}F_2\\B_1\end{bmatrix}
&=
\begin{bmatrix}
-\mathbf{c}_v\\
0\\
\end{bmatrix}\mathbf{s}_w
+
\begin{bmatrix}
1 & d_vZ^{-1}-1\\
Z^{-1} & Z^{-1}
\end{bmatrix}
\begin{bmatrix}F_1\\B_2\end{bmatrix}\\
\begin{bmatrix}F_2\\B_1\end{bmatrix}
&=
\frac{1}{d_vZ^{-1} + 2}
\begin{bmatrix}
-1 & d_v-Z\\
1 & -Z
\end{bmatrix}
\left(
\begin{bmatrix}
-\mathbf{c}_v\\
0\\
\end{bmatrix}\mathbf{s}_w
+
\begin{bmatrix}
1 & d_vZ^{-1}-1\\
Z^{-1} & Z^{-1}
\end{bmatrix}
\begin{bmatrix}F_1\\B_2\end{bmatrix}\\
\right)\\
&=
\frac{1}{d_vZ^{-1} + 2}
\left(
\begin{bmatrix}
\mathbf{c}_v\\
-\mathbf{c}_v\\
\end{bmatrix}\mathbf{s}_w
+
\begin{bmatrix}
2 & Z^{-1}d_v\\
Z^{-1}d_v & 2
\end{bmatrix}
\begin{bmatrix}F_1\\B_2\end{bmatrix}\right)\\
&=
\frac{1}{A d_v + 2 \rho c}\left(
\begin{bmatrix}
\mathbf{c}_v\\
-\mathbf{c}_v\\
\end{bmatrix}\mathbf{s}_w
+
\begin{bmatrix}
2\rho c & A d_v\\
A d_v & 2\rho c
\end{bmatrix}
\begin{bmatrix}
F_1\\B_2
\end{bmatrix}\right)\\
\end{aligned}$$

Now, for the state update equation, use partial pressures as the input

$$
\dot{\mathbf{s}}_v = \mathbf{A}_v \mathbf{s}_v + \mathbf{b}_v\frac{A}{\rho c}(F_1-B_1)
$$

Substitute the output equation for $B_1$:

$$\begin{aligned}
\dot{\mathbf{s}}_v &= \mathbf{A}_v \mathbf{s}_v + \mathbf{b}_v\frac{A}{\rho c}\left[F_1-\left(-\mathbf{c}_v\mathbf{s}_v + \frac{1}{A d_v + 2 \rho c}
\begin{bmatrix}
A d_v & 2\rho c
\end{bmatrix}
\begin{bmatrix}
F_1\\B_2
\end{bmatrix}\right)\right]\\
&= \left[\mathbf{A}_v + \mathbf{b}_v \frac{1}{A d_v + 2 \rho c}\frac{A}{\rho c}\mathbf{c}_v\right]\mathbf{s}_v
+ \mathbf{b}_v\frac{2A}{A d_v + 2 \rho c}
\begin{bmatrix}
1 & -1
\end{bmatrix}
\begin{bmatrix}
F_1\\B_2
\end{bmatrix}\\
\end{aligned}$$

Hence, we have the yielding-wall block:

$$\begin{align}
\mathbf{A} &= \mathbf{A}_v + \mathbf{b}_v \frac{1}{A d_v + 2 \rho c}\frac{A}{\rho c}\mathbf{c}_v\\
\mathbf{B} &= \mathbf{b}_v\frac{2A}{A d_v + 2 \rho c}
\begin{bmatrix}
1 & -1
\end{bmatrix}\\
\mathbf{C} &= \frac{1}{A d_v + 2 \rho c}\begin{bmatrix}1\\-1\\\end{bmatrix}\mathbf{c}_v\\
\mathbf{D} &= \frac{1}{A d_v + 2 \rho c}
\begin{bmatrix}
2\rho c & A d_v\\
A d_v & 2\rho c
\end{bmatrix}
\end{align}$$

## For the loss model in an improper first-order TF 
Now if the series block is represented by an improper first-order transfer function, a la the RL viscous loss model, Story Eq (2.72)...

$$\begin{equation}
P_v = R_v U + L_v \dot{U}
\end{equation}$$

This system does not have a state-space representation, so the above derivation cannot be used. Here, we use $U$ as the state variable $s$, and solve the system of equations for $F_2$, $B_1$, and $\dot{s}$. The governing equations are

$$\begin{align}
F_1 + B_1 &= R_v s + L_v \dot{s} + F_2 + B_2\\
s &= \frac{1}{Z}(F_1-B_1)\\
s &= \frac{1}{Z}(F_2-B_2)\\
\end{align}$$

Manipulate the questions so that $F_2$, $B_1$, and $\dot{s}$ appear on the left hand side and $F_1$, $B_2$, and $s$ appear on the right hand side:

$$\begin{align}
L_v\dot{s} + F_2 - B_1 &= -R_vs+F_1 - B_2\\
\frac{1}{Z}B_1 &= -s + \frac{1}{Z}F_1 \\
\frac{1}{Z} F_2  &= s + \frac{1}{Z}B_2\\
\end{align}$$

Convert the equations to a matrix-vector equation and solve for $[\dot{s}\ F_2\ B_1]^T$:

$$\begin{align}
\begin{bmatrix} 
L_v & 1 & -1\\
0 & 0 & 1/Z\\
0 & 1/Z & 0\\
\end{bmatrix} \begin{bmatrix}\dot{s}\\F_2\\B_1\end{bmatrix} &= \begin{bmatrix}-R_v & 1 & -1\\
-1 & 1/Z & 0\\
1 & 0 & 1/Z\\
\end{bmatrix} \begin{bmatrix}s\\F_1\\B_2\end{bmatrix}\\
\begin{bmatrix}\dot{s}\\F_2\\B_1\end{bmatrix} &= 
\begin{bmatrix}
    1/L_v & Z/L_v & Z/L_v\\
    0 & 0 & Z\\
    0 & Z & 0
\end{bmatrix}
\begin{bmatrix}
    -R_v & 1 & -1\\
    -1 & 1/Z & 0\\
    1 & 0 & 1/Z
\end{bmatrix} \\
\begin{bmatrix}\dot{s}\\F_2\\B_1\end{bmatrix}
&= 
\begin{bmatrix}
    -\frac{R_v+2Z}{L_v} & \frac{2}{L_v} & -\frac{2}{L_v}\\
    Z & 0 & 1\\
    -Z & 1 & 0
\end{bmatrix} 
\begin{bmatrix}s\\F_1\\B_2\end{bmatrix}
\end{align}$$

Accordingly, the resulting statespace coefficients are:

$$\begin{align}
A &= -\frac{R_v+2Z}{L_v} \\
\mathbf{B} &= \frac{2}{L_v} \begin{bmatrix} 1 & -1\end{bmatrix}\\
\mathbf{C} &= Z \begin{bmatrix}1\\-1\end{bmatrix}\\
\mathbf{D} &= \begin{bmatrix}
     0 & 1\\
    1 & 0
\end{bmatrix}\\
\end{align}$$
