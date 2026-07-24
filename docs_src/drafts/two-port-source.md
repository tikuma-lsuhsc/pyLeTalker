Documentation from the doctor on letterhead specific to the life-threatening medical situations and limitations.

# State-space representation of the glottal source

Titze (1984) established the relationship between the supraglott partial pressure $F_2$ and $B_2$ and the glottal flow $U_g$ as

$$
\begin{equation}
F_2 - B_2 = Z_s U_g
\end{equation}
$$

where $Z_s = \rho c / A$  and $A$ is the supraglottal vocal tract area. Accordingly, a stateless state-space model representation of this 2-input, 1-output system is simply

$$
\begin{align}
F_2 &= \begin{bmatrix}
  Z_s & 1
\end{bmatrix}\begin{bmatrix}
  U_g \\ B_2
\end{bmatrix}\\
\end{align}
$$

In other words, only $D$ matrix is present.

## Combining with the next vocal tract subsystem

When this subsystem is attached to a (continuous-time?) )two-port vocal tract model, it creates an algebraic loop because the backward output of the vocal tract ($B_1$) loops back to $F_1$ without any delay. The combined model thus needs to be algebraically connected first.

Let a state-space matrices of the connecting VT subsystem be:

$$
\begin{align}
  \dot{\mathbf{s}} &= \mathbf{A} \mathbf{s}+\begin{bmatrix}
  \mathbf{b}_1 & \mathbf{b}_2
  \end{bmatrix}\begin{bmatrix}
    F_1 \\ B_2
  \end{bmatrix}\\
  \begin{bmatrix}
    F_2\\B_1
  \end{bmatrix}&=\begin{bmatrix}
    \mathbf{c}_1 \\ \mathbf{c}_2
  \end{bmatrix}\mathbf{s}+\begin{bmatrix}
    d_{11} & d_{12}\\
    d_{21} & d_{22}\\
  \end{bmatrix}\begin{bmatrix}
    F_1\\B_2
  \end{bmatrix}  
\end{align}
$$

Connecting $F_2$ of the source to $F_1$ of this subsystem and $B_2$ of the source to $B_1$ of this subsystem expands (4) as

$$
\begin{aligned}
  \dot{\mathbf{s}} &= \mathbf{A} \mathbf{s}+\begin{bmatrix}
  \mathbf{b}_1 & \mathbf{b}_2
  \end{bmatrix}\begin{bmatrix}
    Z_s U_g + B_1 \\ B_2
  \end{bmatrix}\\
  \begin{bmatrix}
    F_2\\B_1
  \end{bmatrix}&=\begin{bmatrix}
    \mathbf{c}_1 \\ \mathbf{c}_2
  \end{bmatrix}\mathbf{s}+\begin{bmatrix}
    d_{11} & d_{12}\\
    d_{21} & d_{22}\\
  \end{bmatrix}\begin{bmatrix}
    Z_s U_g + B_1\\B_2
  \end{bmatrix}  
\end{aligned}
$$

Manipulate the  output equation so that  only appears on the left-hand side:

$$
\begin{aligned}
  B_1 &= \mathbf{c}_2 \mathbf{s}+d_{21}(Z_s U_g + B_1) + d_{22}B_2\\
  (1 - d_{21}) B_1 &= \mathbf{c}_2 \mathbf{s} + d_{21} Z_s U_g + d_{22}B_2\\
  B_1 &= \frac{\mathbf{c}_2}{1 - d_{21}} \mathbf{s} + \frac{d_{21} Z_s}{1 - d_{21}} U_g + \frac{d_{22}}{1 - d_{21}}B_2\\
      &= \tilde{\mathbf{c}}_2 \mathbf{s} + \tilde{d}_{21} U_g + \tilde{d}_{22}B_2
\end{aligned}
$$

Substitute $B_1$ into the state-space equations:

$$
\begin{aligned}
  \dot{\mathbf{s}} &= \mathbf{A} \mathbf{s}+\begin{bmatrix}
  \mathbf{b}_1 & \mathbf{b}_2
  \end{bmatrix}\begin{bmatrix}
    Z_s U_g + \tilde{\mathbf{c}}_2 \mathbf{s} + \tilde{d}_{21} U_g + \tilde{d}_{22}B_2 \\ B_2
  \end{bmatrix}\\
  &= \mathbf{A}\mathbf{s}+\mathbf{b}_1 (Z_s U_g + \tilde{\mathbf{c}}_2 \mathbf{s} + \tilde{d}_{21} U_g + \tilde{d}_{22}B_2) + \mathbf{b}_2 B_2\\
  &= \mathbf{A}\mathbf{s}+\mathbf{b}_1 Z_s U_g + \mathbf{b}_1 \tilde{\mathbf{c}}_2 \mathbf{s} + \mathbf{b}_1 \tilde{d}_{21} U_g + \mathbf{b}_1 \tilde{d}_{22}B_2 + \mathbf{b}_2 B_2\\
  &= (\mathbf{A} + \mathbf{b}_1 \tilde{\mathbf{c}}_2) \mathbf{s} + \begin{bmatrix}
    \mathbf{b}_1 Z_s + \mathbf{b}_1 \tilde{d}_{21} & \mathbf{b}_1 \tilde{d}_{22} +\mathbf{b}_2\end{bmatrix}\begin{bmatrix}
      U_g \\ B_2
    \end{bmatrix}\\
  \begin{bmatrix}
    U_g\\B_2
  \end{bmatrix}&=\begin{bmatrix}
    \mathbf{c}_1 \\ \mathbf{c}_2
  \end{bmatrix}\mathbf{s}+\begin{bmatrix}
    d_{11} & d_{12}\\
    d_{21} & d_{22}\\
  \end{bmatrix}\begin{bmatrix}
    Z_s U_g + \tilde{\mathbf{c}}_2 \mathbf{s} + \tilde{d}_{21} U_g + \tilde{d}_{22}B_2\\B_2
  \end{bmatrix}  
\end{aligned}
$$
