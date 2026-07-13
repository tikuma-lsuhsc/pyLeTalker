# Joining two-port subsystems of wave reflection model

Within a tube segment of wave reflection model, a number of subsystems may be
present in a model. For example, the boundary reflection subsystem plus a yielding 
wall subsystem. As the tube model gets more complex, deriving the system equations
quickly becomes cumbersome. A generalized approach is to express each subsystem
in a state-space representation and devise a way to connect two successive subsystems.

Suppose we have two subsystems with state-space models:

$$\begin{align}
\dot{\mathbf{s}}_1 &= \mathbf{A}_1 \mathbf{s}_1 + \mathbf{B}_1 \mathbf{x}_1 + \mathbf{B}_{\text{aux},1} \mathbf{x}_{\text{aux},1}\\
\mathbf{y}_1 &= \mathbf{C}_1 \mathbf{s}_1 + \mathbf{D}_1 \mathbf{x}_1 + \mathbf{D}_{\text{aux},1} \mathbf{x}_{\text{aux},1}\\
\dot{\mathbf{s}}_2 &= \mathbf{A}_2 \mathbf{s}_2 + \mathbf{B}_2 \mathbf{x}_2 + \mathbf{B}_{\text{aux},2} \mathbf{x}_{\text{aux},2}\\
\mathbf{y}_2 &= \mathbf{C}_2 \mathbf{s}_2 + \mathbf{D}_2 \mathbf{x}_2 + \mathbf{D}_{\text{aux},2} \mathbf{x}_{\text{aux},2}\\
\end{align}$$

where

$$
\mathbf{x}_1 = \begin{bmatrix}F_1\\B_2\end{bmatrix} \quad 
\mathbf{y}_1 = \begin{bmatrix}F_2\\B_1\end{bmatrix} \quad 
\mathbf{x}_2 = \begin{bmatrix}F_2\\B_3\end{bmatrix} \quad 
\mathbf{y}_2 = \begin{bmatrix}F_3\\B_2\end{bmatrix}
$$

and the input and output of the joined system is

$$
\mathbf{x} = \begin{bmatrix}F_1\\B_3\end{bmatrix} \quad 
\mathbf{x}_\text{aux} = \begin{bmatrix}\mathbf{x}_{\text{aux},1}\\\mathbf{x}_{\text{aux},2}\end{bmatrix} \quad 
\mathbf{y} = \begin{bmatrix}F_3\\B_1\end{bmatrix} \quad 
$$

<div style="background-color: aliceblue;">
  <img src="two-port-join-block.svg" width=75%>
</div>

The input and output signals of the state-space equations are sorted to isolate the joining ports ($F_2$ and $B_2$):
$$\begin{align}
\dot{\mathbf{s}}_1 &= \mathbf{A}_1 \mathbf{s}_1 + \begin{bmatrix}\mathbf{b}_{1b} & \tilde{\mathbf{B}}_1\end{bmatrix}\begin{bmatrix}B_2 \\ \tilde{\mathbf{x}}_1\end{bmatrix}\\
\begin{bmatrix}F_2 \\ \tilde{\mathbf{y}}_1\end{bmatrix} &= \begin{bmatrix}\mathbf{c}_{1f} \\ \tilde{\mathbf{C}}_1\end{bmatrix} \mathbf{s}_1 + \begin{bmatrix}d_{1fb} & \mathbf{d}_{1f} \\ \mathbf{d}_{1b} & \tilde{\mathbf{D}}_1 \end{bmatrix}\begin{bmatrix}B_2 \\ \tilde{\mathbf{x}}_1\end{bmatrix}\\
\dot{\mathbf{s}}_2 &= \mathbf{A}_2 \mathbf{s}_2 + \begin{bmatrix}\mathbf{b}_{2f} & \tilde{\mathbf{B}}_2\end{bmatrix}\begin{bmatrix}F_2 \\ \tilde{\mathbf{x}}_2\end{bmatrix}\\
\begin{bmatrix}B_2 \\ \tilde{\mathbf{y}}_2\end{bmatrix} &= \begin{bmatrix}\mathbf{c}_{2b} \\ \tilde{\mathbf{C}}_2\end{bmatrix} \mathbf{s}_2 + \begin{bmatrix}d_{2bf} & \mathbf{d}_{2b} \\ \mathbf{d}_{2f} & \tilde{\mathbf{D}}_2 \end{bmatrix}\begin{bmatrix}F_2 \\ \tilde{\mathbf{x}}_2\end{bmatrix}\\
\end{align}$$
where 
* $\tilde{\mathbf{x}}_1$ and $\tilde{\mathbf{x}}_2$ are all the remaining inputs, 
* $\tilde{\mathbf{y}}_1$ and $\tilde{\mathbf{y}}_2$ are all the remaining outputs, 
* $\mathbf{b}_{1b}$ and $\mathbf{b}_{2f}$ are the column vectors of the connecting signals of the input matrices, 
* $\mathbf{c}_{1c}$ and $\mathbf{c}_{2b}$ are the row vectors of the connecting of the output matrices,
* $d_{2bf}$ is the feedthrough gain from the interfacing input to the interfacing output
* $\mathbf{d}_{1f}$ and $\mathbf{d}_{2b}$ are the row-vector feedthrough gains from the interfacing input to remaining outputs,
* $\mathbf{d}_{1b}$ and $\mathbf{d}_{2f}$ are the column-vector feedthrough gain from the remaining input to interfacing output,
* $\tilde{\mathbf{B}}_k$, $\tilde{\mathbf{C}}_k$, and $\tilde{\mathbf{D}}_k$, $k\in(1,2)$ are the state-space matrices excluding the interfacing input or output.

First, extract and combine the connecting port $[F_2\ B_2]$ from the output equations:
$$
\begin{bmatrix}
F_2\\B_2
\end{bmatrix}
=\begin{bmatrix}
\mathbf{c}_{1f} & \mathbf{0} \\ \mathbf{0} & \mathbf{c}_{2b}
\end{bmatrix}
\begin{bmatrix}
\mathbf{s}_1\\\mathbf{s}_2
\end{bmatrix}
+\begin{bmatrix}
0 & d_{1fb} \\ d_{2bf} & 0
\end{bmatrix}
\begin{bmatrix}
F_2\\B_2
\end{bmatrix}
+\begin{bmatrix}
\mathbf{d}_{1f} & \mathbf{0}\\ \mathbf{0} & \mathbf{d}_{2b}
\end{bmatrix}
\begin{bmatrix}
\tilde{\mathbf{x}}_{1}\\\tilde{\mathbf{x}}_{2}
\end{bmatrix}
$$
Solving for $F_2$ or $B_2$ yields:
$$
\begin{bmatrix}
F_2\\B_2
\end{bmatrix}
=\left(
\mathbf{I} - \begin{bmatrix}0 & d_{1fb} \\ d_{2bf} & 0\end{bmatrix}
\right)^{-1}
\left[
\begin{bmatrix}
\mathbf{c}_{1f} & \mathbf{0} \\ \mathbf{0} & \mathbf{c}_{2b}
\end{bmatrix}
\begin{bmatrix}
\mathbf{s}_1\\\mathbf{s}_2
\end{bmatrix}
+\begin{bmatrix}
\mathbf{d}_{1f} & \mathbf{0}\\ \mathbf{0} & \mathbf{d}_{2b}
\end{bmatrix}
\begin{bmatrix}
\tilde{\mathbf{x}}_{1}\\\tilde{\mathbf{x}}_{2}
\end{bmatrix}
\right]
$$
Compact the notation to
$$
\begin{equation}
\begin{bmatrix}
F_2\\B_2
\end{bmatrix}
=\mathbf{C}_c \mathbf{s}+\mathbf{D}_c\mathbf{x}
\end{equation}
$$
by defining
$$
\begin{align}
\mathbf{C}_c &= \left(
\mathbf{I} - \begin{bmatrix}0 & d_{1fb} \\ d_{2bf} & 0\end{bmatrix}
\right)^{-1}
\begin{bmatrix}
\mathbf{c}_{1f} & \mathbf{0} \\ \mathbf{0} & \mathbf{c}_{2b}
\end{bmatrix} \in \mathbb{R}^{(2\times n_{s})}\\
\mathbf{D}_c &= \left(
\mathbf{I} - \begin{bmatrix}0 & d_{1fb} \\ d_{2bf} & 0\end{bmatrix}
\right)^{-1}\begin{bmatrix}
\mathbf{d}_{1f} & \mathbf{0}\\ \mathbf{0} & \mathbf{d}_{2b}
\end{bmatrix}
\end{align}
$$

Combined the system equations of two subsytems and separate their inputs:
$$
\begin{align}
\dot{\mathbf{s}} 
&= \begin{bmatrix}\mathbf{A}_1 & \mathbf{0}\\\mathbf{0}&\mathbf{A}_2\end{bmatrix} \mathbf{s}
+\begin{bmatrix}
\tilde{\mathbf{B}}_1 & \mathbf{0} \\ \mathbf{0} & \tilde{\mathbf{B}}_2
\end{bmatrix}
\mathbf{x}
+\begin{bmatrix}
\mathbf{0} & \mathbf{b}_{1b} \\ \mathbf{b}_{2f} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
F_2\\B_2
\end{bmatrix}
\\
\mathbf{y} = \begin{bmatrix}\mathbf{\tilde{y}}_2\\\mathbf{\tilde{y}}_1\end{bmatrix}
&=\begin{bmatrix}
\mathbf{0} & \tilde{\mathbf{C}}_2\\
\tilde{\mathbf{C}}_1 & \mathbf{0}\\
\end{bmatrix} \mathbf{s}
+\begin{bmatrix}
\mathbf{0} & \tilde{\mathbf{D}}_2\\
\tilde{\mathbf{D}}_1 & \mathbf{0}\\
\end{bmatrix} \mathbf{x}
+\begin{bmatrix}
\mathbf{d}_{2f} & \mathbf{0}\\
\mathbf{0} & \mathbf{d}_{1b}\\ 
\end{bmatrix}
\begin{bmatrix}F_2\\B_2\end{bmatrix}
\end{align}
$$
Substitution of (9) into (12) and (13) and subsequent simplification yield
$$
\begin{align}
\dot{\mathbf{s}} 
&= \left(\begin{bmatrix}\mathbf{A}_1 & \mathbf{0}\\\mathbf{0}&\mathbf{A}_2\end{bmatrix} 
+\begin{bmatrix}
\mathbf{0} & \mathbf{b}_{1b} \\ \mathbf{b}_{2f} & \mathbf{0}
\end{bmatrix}\mathbf{C}_c \right)\mathbf{s}
+\left(\begin{bmatrix}
\tilde{\mathbf{B}}_1 & \mathbf{0} \\ \mathbf{0} & \tilde{\mathbf{B}}_2
\end{bmatrix}
+\begin{bmatrix}
\mathbf{0} & \mathbf{b}_{1b} \\ \mathbf{b}_{2f} & \mathbf{0}
\end{bmatrix}\mathbf{D}_c\right)\mathbf{x}
\\
\mathbf{y}&=\left(\begin{bmatrix}
\mathbf{0} & \tilde{\mathbf{C}}_2\\
\tilde{\mathbf{C}}_1 & \mathbf{0}\\
\end{bmatrix} 
+\begin{bmatrix}
\mathbf{d}_{2f} & \mathbf{0}\\
\mathbf{0} & \mathbf{d}_{1b}\\ 
\end{bmatrix} \mathbf{C}_c\right) \mathbf{s}
+\left(\begin{bmatrix}
\mathbf{0} & \tilde{\mathbf{D}}_2\\
\tilde{\mathbf{D}}_1 & \mathbf{0}\\
\end{bmatrix} 
+\begin{bmatrix}
\mathbf{d}_{2f} & \mathbf{0}\\
\mathbf{0} & \mathbf{d}_{1b}\\ 
\end{bmatrix}\mathbf{D}_c\right)\mathbf{x}
\end{align}
$$
Accordingly, the state-space matrices of the combined system are
$$
\begin{align}
\mathbf{A} &=\begin{bmatrix}
\mathbf{A}_1 & \mathbf{0}\\
\mathbf{0} & \mathbf{A}_2\\
\end{bmatrix} 
+\begin{bmatrix}
\mathbf{0} & \mathbf{b}_{1b} \\ \mathbf{b}_{2f} & \mathbf{0}
\end{bmatrix} \mathbf{C}_c\\
\mathbf{B} &=\begin{bmatrix}
\tilde{\mathbf{B}}_1 & \mathbf{0}\\
\mathbf{0} & \tilde{\mathbf{B}}_2\\
\end{bmatrix} 
+\begin{bmatrix}
\mathbf{0} & \mathbf{b}_{1b} \\ 
\mathbf{b}_{2f} & \mathbf{0}
\end{bmatrix}\mathbf{D}_c\\
\mathbf{C} &=\begin{bmatrix}
\mathbf{0} & \tilde{\mathbf{C}}_2\\
\tilde{\mathbf{C}}_1 & \mathbf{0}\\
\end{bmatrix} 
+\begin{bmatrix}
\mathbf{d}_{2f} & \mathbf{0}\\\mathbf{0} & \mathbf{d}_{1b}
\end{bmatrix} \mathbf{C}_c\\
\mathbf{D} &=\begin{bmatrix}
\mathbf{0} & \tilde{\mathbf{D}}_2\\
\tilde{\mathbf{D}}_1 & \mathbf{0}\\
\end{bmatrix} 
+\begin{bmatrix}
\mathbf{d}_{2f} & \mathbf{0}\\\mathbf{0} & \mathbf{d}_{1b}
\end{bmatrix}\mathbf{D}_c\\
\end{align}
$$