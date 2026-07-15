# State-space representation of the radiation impedance load

Flanagan estimated the radiation load by a piston in an infinite baffle. This model
represents the termination by a first-order system

$$
\begin{equation}
H_r(s) \triangleq \frac{P_r}{U_r} = \frac{sRL}{R+sL}
\end{equation}
$$

where $P_r$ is the pressure across the impedance and $U_r$ is the flow through
the impedance., and
$$
R = \frac{128 Z}{9\pi^2} \text{ and } L = \frac{8aZ}{3\pi c}.
$$
Here, $A$ and $Z = \rho c/A$ are respectively the cross-sectional area and the characteristic acoustic impedance of the final tube section, and $a = \sqrt{A/\pi}$ is the radius of the piston.

Let a state-space model representation of this system as
$$
\begin{align}
\dot{\mathbf{s}} &= \mathbf{A} \mathbf{s} + \mathbf{b} U\\
P &= \mathbf{c} \mathbf{s} + d U\\
\end{align}
$$
The partial pressures $F$ and $B$ are defined by
$$
\begin{align}
F + B &= P\\
\frac{F - B}{Z} &= U\\
\end{align}
$$
Substituting the output equation and the flow preservation equation into the pressure preservation equation yields
$$
\begin{align}
P &= \mathbf{c} \mathbf{s} + \frac{d}{Z} \left(F-B\right)\\
F + B &= \mathbf{c} \mathbf{s} + \frac{d}{Z} \left(F-B\right)\\
\end{align}
$$
Solve for $B$
$$
\begin{align}
\left(\frac{d}{Z} +1\right)B &= \mathbf{c} \mathbf{s} + \left(\frac{d}{Z} - 1\right)F\\
B &= \frac{\mathbf{c}}{\frac{d}{Z} +1} \mathbf{s} + \frac{\frac{d}{Z} - 1}{\frac{d}{Z} +1}F\\
\end{align}
$$
Hence,
$$
\begin{align}
\begin{bmatrix}
P \\ B\\
\end{bmatrix}
&= 
\frac{1}{\frac{d}{Z} +1}
\begin{bmatrix}
\mathbf{c} \\ \mathbf{c}\\
\end{bmatrix}
\mathbf{s}
+
\frac{1}{\frac{d}{Z} +1}
\begin{bmatrix}
2 \frac{d}{Z}\\
\frac{d}{Z} - 1\\
\end{bmatrix}
F
\end{align}
$$

$$
\begin{equation}\begin{aligned}
(1 + d_rZ^{-1})B &= \mathbf{c}_r \mathbf{s} + (d_rZ^{-1} - 1)F\\
B &= \frac{1}{d_rZ^{-1}+1}\mathbf{c}_r \mathbf{s} + \frac{d_rZ^{-1} - 1}{d_rZ^{-1}+1}F\\
  &= \frac{\rho c}{d_rA+\rho c}\mathbf{c}_r \mathbf{s} + \frac{d_rA - \rho c}{d_rA+\rho c}F\\
\end{aligned}\end{equation}
$$

Then, the radiated pressure is calculated by

$$
\begin{aligned}
P_r &= F+B \\
    &= F + \frac{\rho c}{d_rA+\rho c}\mathbf{c}_r \mathbf{s} + \frac{d_rA - \rho c}{d_rA+\rho c}F\\
    &= \frac{\rho c}{d_rA+\rho c}\mathbf{c}_r \mathbf{s} + \frac{2d_rA}{d_rA+\rho c}F\\  
\end{aligned}
$$

The state update equation in (64) is converted to take $F$ as the input by
substituting (67) and then (68), followed by algebraic simplification:

$$
\begin{equation}\begin{aligned}
\dot{\mathbf{s}} &= \mathbf{A}_r \mathbf{s} + \mathbf{b}_r Z^{-1} \left[F-\left(\frac{1}{d_rZ^{-1}+1}\mathbf{c}_r \mathbf{s} + \frac{d_rZ^{-1} - 1}{d_rZ^{-1}+1}F\right)\right]\\
&= \left[\mathbf{A}_r - \mathbf{b}_r \frac{Z^{-1}}{d_rZ^{-1}+1}\mathbf{c}_r\right] \mathbf{s} + \mathbf{b}_r \left[\frac{2Z^{-1}}{d_rZ^{-1}+1}\right]F\\
&= \left[\mathbf{A}_r - \mathbf{b}_r \frac{A}{d_rA+\rho c}\mathbf{c}_r\right] \mathbf{s} + \mathbf{b}_r \left[\frac{2A}{d_rA+\rho c}\right]F
\end{aligned}\end{equation}
$$

In summary,

$$
\begin{align}
\mathbf{A} &= \mathbf{A}_r - \mathbf{b}_r \frac{A}{d_rA+\rho c}\mathbf{c}_r\\
\mathbf{b} &= \mathbf{b}_r \left[\frac{2A}{d_rA+\rho c}\right]\\
\mathbf{c} &= \frac{1}{d_rZ^{-1}+1}\mathbf{c}_r\\
d &= \frac{d_rA - \rho c}{d_rA+\rho c}\\
\end{align}
$$

Since there is no foward pressure output, appending this block to another reduces
the joined system to be SISO and $\mathbf{B}$, $\mathbf{C}$ and $\mathbf{D}$
matrices in (32)-(34) reduce to

$$
\begin{align}
\mathbf{b} &= 
\frac{1}{\gamma}
\begin{bmatrix}
\mathbf{B}_1 & \mathbf{0}\\
\mathbf{0} & \mathbf{b}_2 \\
\end{bmatrix}
\begin{bmatrix}
\gamma\\
d_{1,11}d_{2,21}\\
d_{1,11}\\
\end{bmatrix}\\
\mathbf{c} &= \frac{1}{\gamma}
\begin{bmatrix}
d_{1,22}d_{2,21} & \gamma &  d_{1,22}\\
\end{bmatrix}
\begin{bmatrix}
\mathbf{C}_1 &0\\
0 & \mathbf{c}_2
\end{bmatrix}\\
d &= \frac{d_{1,22}d_{2,21}d_{1,11}+d_{1,21}}{\gamma}\\
\end{align}
$$
