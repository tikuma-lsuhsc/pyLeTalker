[1] I. Titze, T. Riede, and T. Mau, “Predicting achievable fundamental frequency ranges in vocalization across species,” PLoS Comput Biol, vol. 12, no. 6, p. e1004907, Jun. 2016, doi: 10.1371/journal.pcbi.1004907

### Relationship between fundamental frequency $f_o$ and glottal length $L$

$$
f_o = \frac{1}{2L}\sqrt{\frac{\mu^\prime}{\rho}}
$$

where 

- $\rho$ is the tissue density (typically 1.04 g/cm$^3$) 
- $\mu^\prime$ is the combined shear and tensile stress for vibrational displacement transverse to the string, given by
  
    $$
    \mu^\prime = A e^{B\left(L-L_0\right)/L_0}
    $$

  - $L_0$ is a reference length (male: 16 mm, female: 1.0 mm)
  - $A$ (male: 1.1, female: 2.5)
  - $B$ (male: 16.2, female: 12.9)
  - $\epsilon = (L-L_0)/L_0$ is the strain

Substitute $\mu^\prime$ expression into $f_o(L)$:

$$
\begin{aligned}
f_o &= \frac{1}{2L}\sqrt{\frac{A e^{B\left(L-L_o\right)/L_o}}{\rho}}\\
&= \frac{1}{2L}\sqrt{\frac{A}{\rho} \exp\left[\frac{B\left(L-L_o\right)}{L_o}\right]}\\
&= \frac{1}{2L}\sqrt{\frac{A}{\rho}}\exp\left[\frac{B\left(L-L_o\right)}{2L_o}\right]\\
&= \frac{1}{2L}\sqrt{\frac{A}{\rho}}\exp\left(\frac{BL}{2L_o}\right) \exp\left(- \frac{B}{2}\right)\\
L &= \frac{1}{2f_o}\sqrt{\frac{A}{\rho}} \exp\left(- \frac{B}{2}\right) \exp\left(\frac{BL}{2L_o}\right)\\
\end{aligned}
$$

Use the parametrization shown in [Wikipedia](https://en.wikipedia.org/wiki/Lambert_W_function#Solving_equations): $L = a + b e^{cL}$
$$
\begin{aligned}
a &= 0\\
b &= \frac{1}{2f_o}\sqrt{\frac{A}{\rho}} \exp\left(- \frac{B}{2}\right)\\
c &= \frac{B}{2L_o}\\
\end{aligned}
$$

Then,

$$
L = a - \frac{1}{c}W(-bce^{ac})
$$

where $W(x)$ is the Lambert W function.

$$
\begin{aligned}
f_o &= \frac{1}{2L}\sqrt{\frac{A e^{B\left(L-L_o\right)/L_o}}{\rho}}\\
&= \frac{1}{2L}\sqrt{\frac{A}{\rho} \exp\left[\frac{B\left(L-L_o\right)}{L_o}\right]}\\
&= \frac{1}{2L}\sqrt{\frac{A}{\rho}}\exp\left[\frac{B\left(L-L_o\right)}{2L_o}\right]\\
&= \frac{1}{2L}\sqrt{\frac{A}{\rho}}\exp\left(\frac{BL}{2L_o}\right) \exp\left(- \frac{B}{2}\right)\\
L \exp\left(-\frac{B}{2L_o}L\right) &= \frac{1}{2f_o} e^{-B/2} \sqrt{\frac{A}{\rho}}\\
\end{aligned}
$$

Let 

$$
\begin{aligned}
\alpha &= -\frac{B}{2L_o}\\
\beta &= \frac{1}{2 f_o} e^{B/2} \sqrt{\frac{A}{\rho}}\\
\end{aligned}
$$

Then, we have

$$
L e^{\alpha L} = \beta
$$

Define $x = \alpha L$ then

$$
x e^{x} = \alpha \beta
$$

which is the Lambert W function and can be solved numerically for $x$ then $L=x/\alpha$.


# Titze 84 derivation

## Equation (31)
$$\begin{aligned}
\frac{1}{2}|u|u + \frac{c a^2}{k A^*} u - \frac{2 a^2 P_\Delta}{k\rho}  &= 0\\
\end{aligned}
$$
where $\Delta_P = P_1^+-P_2^-$

Let
$$\begin{aligned}
b &= \frac{c a^2}{k A^*} \ge 0\\
c &= - \frac{2 a^2 P_\Delta}{k\rho}\\
\end{aligned}$$

$$
\frac{1}{2}|u|u + b u + c = 0
$$

2 cases:

$$
\begin{cases}
\frac{1}{2} u^2 + b u + c  = 0, &\text{if } u\ge0\\
-\frac{1}{2} u^2 + b u + c = 0, &\text{if } u<0\\
\end{cases}
$$

Solve the equations
$$
u = \begin{cases}
(-b \pm \sqrt{b^2 - 2 c}), &\text{if } u\ge0\\
(b \mp \sqrt{b^2 + 2 c}), &\text{if } u<0\\
\end{cases}
$$


$$
b^2 - 2c = \frac{c^2 a^4}{k^2 A^{*2}} \pm \frac{4 a^2 P_\Delta}{k\rho}
$$
