# State-space representation of the radiation impedance load

Flanagan estimated the radiation load by a piston in an infinite baffle. This model represents the termination of a wave-reflection two-port transmission-line model by a first-order LTI subsystem
$$
\begin{equation}
H(s) \triangleq \frac{P}{U} = \frac{sRL}{R+sL}
\end{equation}
$$
where $P$ is the pressure across the impedance and $U$ is the flow through
the impedance, 
$$
R = \frac{128 Z}{9\pi^2} \text{ and } L = \frac{8aZ}{3\pi c}.
$$
Here, $A$ and $Z = \rho c/A$ are respectively the cross-sectional area and the characteristic acoustic impedance of the final tube section, and $a = \sqrt{A/\pi}$ is the radius of the piston.

Let a state-space model representation of this system as
$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A}_r \mathbf{s} + \mathbf{b}_r U\\
P &= \mathbf{c}_r \mathbf{s} + d_r U\\
\end{align}
$$
(We generically assume unknown number of states though (1) is a first-order system.) The partial pressures $F$ and $B$ are defined by
$$
\begin{align}
F + B &= P\\
\frac{F - B}{Z} &= U\\
\end{align}
$$
Substituting the flow preservation equation (5) into the output equation (3)  then into the pressure preservation equation (4) yields
$$
\begin{align}
F + B &= \mathbf{c}_r \mathbf{s} + \frac{d_r}{Z} \left(F-B\right)\\
\end{align}
$$
Gather $B$ to the left-hand side:
$$
\begin{equation}
\left(\frac{d_r}{Z} +1\right)B = \mathbf{c}_r \mathbf{s} + \left(\frac{d_r}{Z} - 1\right)F
\end{equation}
$$
Manipulate the pressure preservation equation (4) so that the $P$ and $B$ are on the left-hand side, and combine it with (7) as a vector-matrix equation:
$$
\begin{bmatrix}0 & \frac{d_r}{Z}+1\\1 & -1\end{bmatrix}
\begin{bmatrix}P\\B\end{bmatrix} 
=\begin{bmatrix}\mathbf{c}_r \\ \mathbf{0}\end{bmatrix}\mathbf{s} 
+\begin{bmatrix}\frac{d_r}{Z} - 1 \\ 1\end{bmatrix}F
$$
Let
$$
\mathbf{Q} = \begin{bmatrix}0 & \frac{d_r}{Z}+1\\1 & -1\end{bmatrix}
$$
so that the output and feedthrough matrices of the two-port system are given by
$$
\begin{align}
\mathbf{C} &= \mathbf{Q}^{-1}\begin{bmatrix}\mathbf{c}_r\\\mathbf{0}\end{bmatrix} \in \mathbb{R}^{2 \times n_{st}}\\
\mathbf{d} &= \mathbf{Q}^{-1}\begin{bmatrix}\frac{d_r}{Z} - 1 \\ 1\end{bmatrix} \in \mathbb{R}^2
\end{align}
$$
To convert the state equation, substitute (5) into (2):
$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A}_r \mathbf{s} + \mathbf{b}_r \frac{F - B}{Z}\\
 &= \mathbf{A}_r \mathbf{s} + \mathbf{b}_r \frac{F - (\mathbf{c}_0 \mathbf{s} + d_0 F)}{Z}\\
\end{align}
$$
where $\mathbf{c}_0$ is the first row of $\mathbf{C}$ in (8) and $d_0$ is the first element of $\mathbf{d}$ in (9), respectively.
The final simplification yields
$$
\begin{align}
\mathbf{A} &= \mathbf{A}_r - \mathbf{b}_r\frac{1}{Z}\mathbf{c}_0 \in \mathbb{R}^{n_{st} \times n_{st}}\\
\mathbf{b} &= \mathbf{b}_r \left(1 - d_0\right) \frac{1}{Z} \in \mathbb{R}^{n_{st}}\\
\end{align}
$$
