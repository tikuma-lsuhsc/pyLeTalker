## vocal tract signal propagation

Inputs: $f_{1,n}$ & $b_{K,n}$

Outputs: $f_{K,n}$ & $b_{1,n}$

For $k = 1, ...$
$$
\begin{bmatrix}
    \tilde{f}_{k+1,n}\\
    \tilde{b}_{k,n}
\end{bmatrix} = 
\begin{bmatrix}
    \alpha & r\\
    -r & \alpha
\end{bmatrix}
\begin{bmatrix}
    f_{k,n}\\
    b_{k+1,n}
\end{bmatrix}
$$

    % ---even junctions [(F2,B3),(F4,B5),...]->[(F3,B2),(F5,B4),...]--- */
    % ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */

$$
f_{k,n} = \begin{cases}
\alpha \left[ \alpha\ b_{1,n-1} +  \frac{\rho c}{a_1}\ u_{g,n}\right] & k=1\\
(r_k+1) f_{k-1,n} - r_k b_{k,n},  & k = 2, 4, ..., K\\
\alpha \left( (r_k+1) f_{k-1,n-1}  - r_k b_{k,n-1} \right),  & k = 3, 5, ..., K-1\\
\end{cases}
$$

$$
b_{k,n} = \begin{cases}
r_k f_{k,n}  - (r_k-1) b_{k+1,n},  & k = 1, 3, ..., K-1\\
\alpha_k \left( r_k f_{k,n-1}  - (r_k-1) b_{k+1,n-1} \right),  & k = 2, 4, ..., K-2\\
\alpha_k \left[ \left(a_2 f_{K,n-1} + a_1 f_{K,n-2} + b_1 b_{K,n-1}\right)/b_2\right] & k=K \\
\end{cases}
$$


$K=6$, $M=3$
$$
\begin{align*}
f_{2,n} &= (r_1+1) \alpha_1 f_{1,n} - r_1 \alpha_2 b_{2,n} \\
f_{4,n} &= (r_3+1) \alpha_3 f_{3,n} - r_3 \alpha_4 b_{4,n} \\
f_{6,n} &= (r_5+1) \alpha_5 f_{5,n} - r_5 \alpha_6 b_{6,n} \\
b_{1,n} &= r_1 \alpha_1 f_{1,n} - (r_1-1) \alpha_2 b_{2,n}\\
b_{3,n} &= r_3 \alpha_3 f_{3,n} - (r_3-1) \alpha_4 b_{4,n}\\
b_{5,n} &= r_5 \alpha_5 f_{5,n} - (r_5-1) \alpha_6 b_{6,n}\\
\end{align*}
$$

$$
\begin{align*}
\begin{bmatrix}
f_{2,n} \\
f_{4,n} \\
b_{3,n} \\
b_{5,n} \\
\end{bmatrix} &=
\begin{bmatrix}
 0 & 0 & -r_1 \alpha_2 & 0\\
 (r_3+1) \alpha_3 & 0 & 0 & - r_3 \alpha_4 \\
 r_3 \alpha_3 & 0 & 0 & (1-r_3) \alpha_4 \\
 0 & r_5 \alpha_5 & 0 & 0  \\
\end{bmatrix} \begin{bmatrix}
f_{3,n} \\
f_{5,n} \\
b_{2,n} \\
b_{4,n} \\
\end{bmatrix} + \begin{bmatrix}
(r_1+1) \alpha_1 & 0 \\
0 & 0 \\
0 & 0\\
0 & (1-r_5) \alpha_6
\end{bmatrix} \begin{bmatrix}f_{1,n} \\b_{6,n} \\\end{bmatrix} \\
\end{align*}
$$
$$
\begin{align*}
\begin{bmatrix}
f_{6,n} \\
b_{1,n} \\
\end{bmatrix} &=
\begin{bmatrix}
 0 & (r_5+1) \alpha_5 & 0 & 0  \\
 0 & 0 & (1-r_1) \alpha_2 & 0\\
\end{bmatrix} \begin{bmatrix}
f_{3,n} \\
f_{5,n} \\
b_{2,n} \\
b_{4,n} \\
\end{bmatrix} + \begin{bmatrix}
0 & - r_5 \alpha_6 \\
r_1 \alpha_1 & 0 \\
\end{bmatrix} \begin{bmatrix}f_{1,n} \\b_{6,n} \\\end{bmatrix} \\
\end{align*}
$$


$$
\begin{align*}
f_{3,n} &= (r_2+1) \alpha_2 f_{2,n-1}  - r_2 \alpha_3 b_{3,n-1}\\
f_{5,n} &= (r_4+1) \alpha_4 f_{4,n-1}  - r_4 \alpha_5 b_{5,n-1}\\
b_{2,n} &= r_2 \alpha_2 f_{2,n-1}  - (r_2-1) \alpha_3 b_{3,n-1}\\
b_{4,n} &= r_4 \alpha_4 f_{4,n-1}  - (r_4-1) \alpha_5 b_{5,n-1}\\
\end{align*}
$$

$$
\begin{align*}
\begin{bmatrix}
f_{3,n} \\
f_{5,n} \\
b_{2,n} \\
b_{4,n} \\
\end{bmatrix} &=
\begin{bmatrix}
 (r_2+1) \alpha_2 & 0  & -r_2\alpha_3 & 0 \\
 0 & (r_4+1) \alpha_4 &  0 & -r_4 \alpha_5 \\
 r_2\alpha_2 & 0  & (1-r_2)\alpha_3 & 0 \\
 0 & r_4\alpha_4  & 0 & (1-r_4)\alpha_5 \\
\end{bmatrix} \begin{bmatrix}
f_{2,n-1} \\
f_{4,n-1} \\
b_{3,n-1} \\
b_{5,n-1} \\
\end{bmatrix}\\
\end{align*}
$$
## Vector-Matrix Notation of Wave Reflection Model

Inputs: 
$$
\mathbf{x}_n = \begin{bmatrix}f_{1,n} & b_{K,n}\end{bmatrix}^T
$$

Define the outputs of the "odd" junctions as the states: $f_{k,n}, b_{k-1,n}$, $k=2, 4, ..., K$ (outputs of odd junctions)

$$
\mathbf{s}_n = \begin{bmatrix}
\mathbf{f}_n & \mathbf{b}_n \\
\end{bmatrix}^T
$$
where
$$
\mathbf{f}_n = \begin{bmatrix}
f_{2,n} & f_{4,n} & \cdots & f_{K, n}
\end{bmatrix}^T \in \mathbb{R}^{K}
$$
and
$$
\mathbf{b}_n = \begin{bmatrix}
b_{1,n} & b_{3,n} & \cdots & b_{K-1, n}
\end{bmatrix}^T \in \mathbb{R}^{K}
$$
It is important to be aware that $\mathbf{f}_n$ and $\mathbf{b}_n$ are at the respective input end of the tube sections and thus have not yet attenuated by the $\alpha$ factor.

Then we can rewrite the "odd junction" update equations in vector-matrix format as follows.
$$
\begin{equation}
\begin{bmatrix} 
\mathbf{f}_n \\ 
\mathbf{b}_n 
\end{bmatrix} = 
\begin{bmatrix}
\mathbf{R}_{ff} & \mathbf{R}_{fb} \\
\mathbf{R}_{bf} & \mathbf{R}_{bb}
\end{bmatrix}
\underbrace{\begin{bmatrix}
\mathbf{\tilde{A}}_f & \mathbf{0} \\ 
\mathbf{0} & \mathbf{\tilde{A}}_b
\end{bmatrix}}_{\text{currently }=\mathbf{0}}
\begin{bmatrix}
\tilde{\mathbf{f}}_n \\ 
\tilde{\mathbf{b}}_n
\end{bmatrix}
+
\begin{bmatrix}
\mathbf{G}_f\\
\mathbf{G}_b
\end{bmatrix}
\underbrace{
\begin{bmatrix}
\alpha_1 & 0\\
0 & \alpha_K
\end{bmatrix}}_\text{trachea $a_K=0$}
\mathbf{x}_n
\end{equation}
$$
where
$$
\begin{align}
\tilde{\mathbf{f}}_n &= \begin{bmatrix}
f_{3,n} & f_{5,n} & \cdots & f_{K-1, n}
\end{bmatrix}^T  \in \mathbb{R}^{K-1}\\
\tilde{\mathbf{b}}_n &= \begin{bmatrix}
b_{2,n} & b_{4,n} & \cdots & b_{K-2, n}
\end{bmatrix}^T \in \mathbb{R}^{K-1}
\end{align}
$$
are the intermediate signals (the outputs of the even junctions), and
$$
\begin{align}
\mathbf{\tilde{A}}_f &= \begin{bmatrix}
\alpha_{3} & && 0\\
& \alpha_{5} & &&\\
 & & \ddots &  \\
0 & & &\alpha_{K-1} \\
\end{bmatrix} \in \mathbb{R}^{(K/2-1) \times (K/2-1)}\\
\mathbf{\tilde{A}}_b &= \begin{bmatrix}
\alpha_{2} & && 0\\
& \alpha_{4} & &&\\
 & & \ddots &  \\
0 & & &\alpha_{K-2} \\
\end{bmatrix} \in \mathbb{R}^{(K/2-1) \times (K/2-1)}\\
\mathbf{R}_{ff} &= \begin{bmatrix}
0  & 0 &\cdots & 0 \\
1+r_3 & 0&\cdots & 0  \\
0 & 1+r_5 & & &   \\
 & \ddots &   \\
0 &  & 1+r_{K-1}  
\end{bmatrix} \in \mathbb{R}^{(K/2) \times (K/2-1)} \\
\mathbf{R}_{bf} = -\mathbf{R}_{fb} &= \begin{bmatrix}
r_1  & 0 & \cdots & 0 \\
0 &r_3 &  &  \\
\vdots & & \ddots & \vdots  \\
0 & \cdots & 0 & r_{K-1}
\end{bmatrix}  \in \mathbb{R}^{(K/2) \times (K/2)} \\ \\
\mathbf{R}_{bb} &= \begin{bmatrix}
1+r_1 & 0 & \cdots & 0  \\
0 & 1+r_3 & & 0  \\
\vdots & & \ddots & & \\
0 & \cdots & 0 & 1+r_{K-1}   \\
0  & \cdots & 0 & 0   \\
\end{bmatrix} \in \mathbb{R}^{(K/2) \times (K/2-1)} \\
\mathbf{G}_f &= \begin{bmatrix}
1+r_1 &0 \\
0 &0 \\
\vdots &\vdots\\
0 & -r_{K-1} \\
\end{bmatrix} \in \mathbb{R}^{(K/2) \times 2}\\
\mathbf{G}_b &= \begin{bmatrix}
r_1 &0 \\
0 & 0 \\
\vdots &\vdots\\
0 & 1-r_{K-1} \\
\end{bmatrix} \in \mathbb{R}^{(K/2) \times 2}\\
\end{align}
$$

The intermediate signals (or the outputs of the even junctions) $\mathbf{\tilde{f}}_n$ and $\mathbf{\tilde{b}}_n$ are calculated from the previous states:
$$
\begin{align}
\mathbf{\tilde{f}}_n
&= 
\begin{bmatrix}
\mathbf{\tilde{R}}_{ff} & \mathbf{\tilde{R}}_{fb}
\end{bmatrix}
\begin{bmatrix}
\mathbf{A}_f & \mathbf{0}\\
\mathbf{0} & \mathbf{A}_b
\end{bmatrix}
\begin{bmatrix}
\mathbf{f}_{n-1}\\\mathbf{b}_{n-1}\end{bmatrix}\\
\mathbf{\tilde{b}}_n
&= 
\begin{bmatrix}
\mathbf{\tilde{R}}_{bf} & \mathbf{\tilde{R}}_{bb}
\end{bmatrix}
\begin{bmatrix}
\mathbf{A}_f & \mathbf{0}\\
\mathbf{0} & \mathbf{A}_b
\end{bmatrix}
\begin{bmatrix}\mathbf{f}_{n-1}\\\mathbf{b}_{n-1}\end{bmatrix}\\
\end{align}
$$

$$
\begin{align}
\mathbf{A}_f &= \begin{bmatrix}
\alpha_{2} & && 0 & 0\\
& \alpha_{4} & &&\\
 & & \ddots &  \\
0 & & &\alpha_{K-2} & 0 \\
\end{bmatrix} \in \mathbb{R}^{(K/2-1) \times (K/2)}\\
\mathbf{A}_b &= \begin{bmatrix}
\alpha_{1} & && 0\\
& \alpha_{3} & &&\\
 & & \ddots &  \\
0 & & &\alpha_{K-1} \\
\end{bmatrix} \in \mathbb{R}^{(K/2-1) \times (K/2-1)}\\
\mathbf{\tilde{R}}_{ff} &= \begin{bmatrix}
1+r_3 & && 0\\
& 1+r_5 & &&\\
 & & \ddots &  \\
0 & & &1+r_{K-1} \\
\end{bmatrix} \in \mathbb{R}^{(K/2-1) \times (K/2-1)} \\
-\mathbf{\tilde{R}}_{fb} = \mathbf{\tilde{R}}_{bf} &= \begin{bmatrix}
r_3  & 0 & \cdots & 0 \\
0 &r_5 &  &  \\
\vdots & & \ddots & \vdots  \\
0 & \cdots & 0 & r_{K=1}
\end{bmatrix}  \in \mathbb{R}^{(K/2-1) \times (K/2-1)} \\ 
 \mathbf{\tilde{R}}_{bb} &= \begin{bmatrix}
1-r_3 & && 0\\
& 1-r_5 & &&\\
 & & \ddots &  \\
0 & & &1-r_{K-1} \\
\end{bmatrix} \in \mathbb{R}^{(K/2-1) \times (K/2-1)} \\
\end{align}
$$

Substituting (11) and (12) into (1) yields

$$
\begin{equation}
\begin{bmatrix} 
\mathbf{f}_n \\ 
\mathbf{b}_n 
\end{bmatrix} = 
\begin{bmatrix}
\mathbf{R}_{ff} & -\mathbf{R}_{bf} \\
\mathbf{R}_{bf} & \mathbf{R}_{bb}
\end{bmatrix}
\begin{bmatrix}
\mathbf{\tilde{A}}_f & \mathbf{0} \\
\mathbf{0} & \mathbf{\tilde{A}}_b
\end{bmatrix}
\begin{bmatrix}
\mathbf{\tilde{R}}_{ff} & -\mathbf{\tilde{R}}_{bf} \\
\mathbf{\tilde{R}}_{bf} & \mathbf{\tilde{R}}_{bb}
\end{bmatrix}
\begin{bmatrix}
\mathbf{A}_f & \mathbf{0} \\
\mathbf{0} & \mathbf{A}_b
\end{bmatrix}
\begin{bmatrix}
\mathbf{f}_{n-1} \\ 
\mathbf{b}_{n-1}
\end{bmatrix}
+
\begin{bmatrix}
\mathbf{G}_f\\
\mathbf{G}_b
\end{bmatrix}
\begin{bmatrix}
\alpha_1 & 0\\
0 & \alpha_{K}
\end{bmatrix}
\mathbf{x}_n
\end{equation}
$$

This can be notated in the standard linear state-space notation:
$$
\mathbf{s}_n = \mathbf{A} \mathbf{s}_{n-1} + \mathbf{B} \mathbf{x}_n
$$
where
$$
\mathbf{A} = 
\begin{bmatrix}
\mathbf{R}_{ff} & -\mathbf{R}_{bf} \\
\mathbf{R}_{bf} & \mathbf{R}_{bb}
\end{bmatrix}
\begin{bmatrix}
\mathbf{\tilde{A}}_f & \mathbf{0} \\
\mathbf{0} & \mathbf{\tilde{A}}_b
\end{bmatrix}
\begin{bmatrix}
\mathbf{\tilde{R}}_{ff} & -\mathbf{\tilde{R}}_{bf} \\
\mathbf{\tilde{R}}_{bf} & \mathbf{\tilde{R}}_{bb}
\end{bmatrix}
\begin{bmatrix}
\mathbf{A}_f & \mathbf{0} \\
\mathbf{0} & \mathbf{A}_b
\end{bmatrix}
$$
and 
$$
\mathbf{b} = 
\begin{bmatrix}
\mathbf{G}_f\\
\mathbf{G}_b
\end{bmatrix}
\begin{bmatrix}
\alpha_1 & 0\\
0 & \alpha_{K}
\end{bmatrix}
$$

To connect to this system, define the external inputs as
$$
\begin{align*}
f_{1,n} &= f_{\text{in},n} \\
b_{K,n} & = b_{\text{out},n}
\end{align*}
$$
and the outputs as
$$
\mathbf{y}_n =
\begin{bmatrix}
b_{\text{in},n} \\
f_{\text{out},n} \\
\end{bmatrix}=
\begin{bmatrix}
\alpha_1 \mathbf{e}_{b_1}\\
\alpha_K \mathbf{e}_{f_K}\\
\end{bmatrix}
\mathbf{s}_{n-1}
$$
where $\mathbf{e}_{p}$ is all zero-vector except for one at the pressure term associated with $p$. The $\alpha$ multipliers account for the propagation of the pressure signals to through the last tube section.

## Vocal-Tract (+Lip Radiation) Vector-Matrix Formulation

The vocal tract takes the glottal flow $u_{g,n}$ as the input and produces the radiated acoustic signal $p_{o,n}$ as well as the forward and backward epiglottic pressure signals, $f_{e,n} = f_{\text{in},n}$ and $b_{e,n} = b_{\text{in},n}$. 

At the input end, we assume that the backward propagating wave (after the tube attenuation) reflects losslessly at the supralaryngeal surface. Thus, the system converting the glottal flow to the forward pressure input takes the form

$$
f_{\text{in},n} = b_{\text{in},n-1} +  \gamma_1u_{g,n}
$$

where $\gamma_1 = \rho c/a_1$. Combining this to the base state update equation yields:

$$
\begin{bmatrix}
f_{1,n}\\
\mathbf{s}_n \\
\end{bmatrix} = \begin{bmatrix}0 & \alpha_1 \mathbf{e}_{b_1}\\ \mathbf{b}_1 & \mathbf{A}\end{bmatrix} \begin{bmatrix}
f_{1,n-1}\\
\mathbf{s}_{n-1} \\
\end{bmatrix} + \begin{bmatrix}
\gamma_1 & \mathbf{b}_2
\end{bmatrix} \begin{bmatrix}
\ u_{g,n} \\
b_{\text{out},n} \\
\end{bmatrix}
$$

Next, the Flanagan's lip radiation model can be expressed by the state-space state update equation:

$$
\begin{align*}
\begin{bmatrix}
b_{\text{out},n}\\
f_{\text{out},n}\\
p_{o,n}\\
\end{bmatrix}&=
\begin{bmatrix}
a_1/b_2 & a_2/b_2 & 0\\
0 & 0 & 0\\
(a_1-b_1)/b_2 & 0 & b_1/b_2\\
\end{bmatrix}
\begin{bmatrix}
b_{\text{out},n-1}\\
f_{\text{out},n-1}\\
p_{o, n-1}\\
\end{bmatrix}+
\begin{bmatrix}
a_2/b_2\\
1\\
1+a_2/b_2\\
\end{bmatrix} \alpha_K f_{K,n} \\
\mathbf{s}_{\text{lip},n} &=
\mathbf{A}_\text{lip}
\mathbf{s}_{\text{lip},n-1} +
\mathbf{b}_\text{lip}
f_{K,n}
\end{align*}
$$


$$
\begin{bmatrix}
f_{1,n}\\
\mathbf{s}_n \\
\mathbf{s}_{\text{lip},n} \\
\end{bmatrix} = \begin{bmatrix}
0 & \alpha_1 \mathbf{e}_{b_1} & 0 \\ 
\mathbf{b}_1 & \mathbf{A} & \mathbf{b}_2|\mathbf{0} \\
\mathbf{0} & \mathbf{0}|\mathbf{b}_\text{lip}|\mathbf{0} & \mathbf{A}_\text{lip}
\end{bmatrix} \begin{bmatrix}
f_{1,n-1}\\
\mathbf{s}_{n-1} \\
\mathbf{s}_{\text{lip}, n-1} \\
\end{bmatrix} + \gamma_1 \ u_{g,n}
$$
