## AM Modulation

$$
\begin{aligned}
x(t) &= A \left[(1-B) + B \cos \left(\omega_{AM} t + \phi_1\right)\right] \sin \left[ \omega_0 t + \phi_0\right]\\
&= \left[1 + \frac{B}{1-B} \cos \left(\omega_{AM} t + \phi_1\right)\right] \sin \left[ \omega_0 t + \phi_0\right] \text{ if } A=\frac{1}{1-B}
\end{aligned}
$$

## ~~Condition 1: unit-gain peaks the first time~~

$$
\begin{aligned}
\omega_0 t + \phi_0 &= \pi/2\\
t_\text{peak}  &= \frac{\pi/2 - \phi_0}{\omega_0}\\
\end{aligned}
$$

$$
\begin{aligned}
x(t_\text{peak}) &= \left[(1-B) + B \cos \left(\omega_{AM} t_\text{peak} + \phi_1\right)\right]  = 1\\
\cos \left(\omega_{AM} t_\text{peak} + \phi_1\right) &= 1\\
\phi_1 &= -\omega_{AM} t_\text{peak} = \frac{\omega_{AM}}{\omega_0}(\phi_0-\pi/2) = \frac{f_{AM}}{f_0}(\phi_0-\pi/2)
\end{aligned}
$$

## Condition 2: AM extent, $\epsilon = (A_\text{max}-A_\text{min})/(A_\text{max}+A_\text{min})$

$A_\text{max}$ and $A_\text{min}$ occur when the modulated sinusoid peaks ($=B$) with the carrier sinusoid and when it negatively peaks ($=-B$) assuming that the peaks align.

$$
\begin{aligned}
A_\text{max} &= (1-B) + B = 1\\
A_\text{min} &= (1-B) - B = 1-2B\\
\end{aligned}
$$

Substitute into the $\epsilon$:

$$
\begin{aligned}
\epsilon &= (1-(1-2B))/(1+(1-2B))\\
         &= B/(1-B)\\
\end{aligned}
$$

Solve for $B$:

$$
\begin{aligned}
\epsilon-\epsilon B &= B\\
B(1+\epsilon) &= \epsilon\\
B &= \frac{\epsilon}{1+\epsilon}
\end{aligned}
$$



## ~~Condition 3: Maintain average power = 0.5~~

$$
\begin{aligned}
Average Power &= \lim_{T→∞}\int_0^T x^2(t) dt \\
&= A^2\left(\frac{3}{4} B^2 - B + \frac{1}{2}\right) = \frac{1}{2}\\
A &= \sqrt{\frac{2}{3 B^2 - 4 B + 2}}\\
\end{aligned}
$$

## Condition 4: Limit $\phi_1$ so that the $A_\text{max}$ and $A_\text{min}$ are guaranteed to be hit under entrainment

# FM Modulation

$$
\begin{aligned}
x(t) &= A \sin \left[ \left\{\int_0^t {\omega_0 + B \cos \left(\omega_{FM} \tau + \phi_1\right) d\tau}\right\} + \phi_0\right]\\
&= A \sin \left[ \omega_0 t + B \int_0^t {\cos \left(\omega_{FM} \tau + \phi_1\right) d\tau} + \phi_0\right]\\
&= A \sin \left[ \omega_0 t + \frac{B}{\omega_{FM}} \sin \left(\omega_{FM} t + \phi_1\right)  + \phi_0\right]\\
\end{aligned}
$$

instantaneous frequency:

$$
\omega(t) = \omega_0 + B \cos \left(\omega_{FM} t + \phi_1 \right)
$$

## Condition 1: max frequency when the first sine wave peaks

$$
\begin{aligned}
\omega_0 t + \phi_0 &= \pi/2\\
t_\text{peak}  &= \frac{\pi/2 - \phi_0}{\omega_0}\\
\end{aligned}
$$

$$
\begin{aligned}
\omega_{FM} t_\text{peak} + \phi_1 &= 0 \\
\phi_1 &= -\omega_{FM} t_\text{peak}\\
 &= \frac{\omega_{FM}}{\omega_0} \left(\phi_0 - \frac{\pi}{2}\right)\\
 &= \frac{f_{FM}}{f_0} \left(\phi_0 - \frac{\pi}{2}\right)\\
\end{aligned}
$$

## Condition 2: FM extent, $\epsilon = (T_\text{max}-T_\text{min})/(T_\text{max}+T_\text{min})$

Peak Instantaneous frequencies:

$$
\begin{aligned}
\epsilon &= \frac{T_\text{max}-T_\text{min}}{T_\text{max}+T_\text{min}}\\
         &= \frac{1/F_\text{min}-1/F_\text{max}}{1/F_\text{min}+1/F_\text{max}}\\
         &= \frac{F_\text{max} - F_\text{min}}{F_\text{min} + F_\text{max}}\\
\end{aligned}
$$

$$
\begin{aligned}
\max f_0(t) &= f_0 + B/2\pi &\rightarrow \ T_\text{min} = 1/(f_0 + B/2\pi)\\
\min f_0(t) &= f_0 - B/2\pi &\rightarrow \ T_\text{max} = 1/(f_0 - B/2\pi)\\\\
\end{aligned}
$$

Substitute them into $\epsilon$

$$
\begin{aligned}
\epsilon &= \frac{(f_0 - B/2\pi)-(f_0 + B/2\pi)}{(f_0 - B/2\pi)(f_0 + B/2\pi)} / \frac{(f_0 - B/2\pi) + (f_0 + B/2\pi)}{(f_0 - B/2\pi)(f_0 + B/2\pi)}\\
         &= -\frac{B}{2 \pi f_0}\\
B &= 2 \pi f_0 \epsilon
\end{aligned}
$$

## Solve integral

$$
\begin{aligned}
x(t) &= A \sin \left[ \omega_0 t + \frac{B}{\omega_{FM}} \sin \left(\omega_{FM} t + \phi_1\right)\right]\\
&= A \sin \left[ \omega_0 t + \epsilon\frac{f_0}{f_{FM}} \sin \left(\omega_{FM} t + \frac{f_{FM}}{f_0} \left(\phi_0 - \frac{\pi}{2}\right) \right)\right]\\
\end{aligned}
$$

# Approximation of Time-Varying FM Cases

$$
\begin{aligned}
x(t) &= A \sin \left[ \int_{-\infty}^t {\omega_0 (\tau) + B(\tau) \cos \phi_M(\tau) d\tau}\right]\\
&= A \sin \left[ \phi_o(t) + \int_{-\infty}^t \omega_o(\tau) \epsilon(\tau) \cos r_M(\tau) \phi_o( \tau) d\tau\right]\\
\end{aligned}
$$

The integral has closed-form solutions when its parameters are either constants or simple functions. To approximate the integral when all the parameters to be arbitrarily time-varying, we turn to Taylor series, it is of the foremost importance to keep the periodicity of $\cos(\phi_M(t))$ we turn to Taylor series expansion. Let the integral term be
$$
\zeta_{FM}(t) \triangleq \omega_o(\tau) \epsilon(\tau) \cos r_M(\tau) \phi_o( \tau).
$$
Its Taylor series expansion is then given by
$$
\zeta_{FM}(t) = \sum_{n=0}^\infty \frac{\zeta_{FM}^{(n)}(t_0)}{n!}(t-t_0)^n.
$$
Then, the phase function of the frequency modulation $\phi_{FM}(t)$ is the antiderivative of $\zeta_{FM}(t)$, which can also be represented as a series:
$$
\begin{aligned}
\phi_{FM}(t) \triangleq \int{\zeta_{FM}(t) dt} &= \sum_{n=0}^\infty \frac{\zeta_{FM}^{(n)}(t_0)}{(n+1)!}(t-t_0)^{n+1}\\
\end{aligned}
$$

To best approximate $\phi_{FM}(t)$, it is crucial for the zero-crossings of the cosine term, $s(t) \triangleq \cos r_M(t)\phi_o(t)$, to be accurate so that the modulator frequency is accurately represented. Meanwhile, approximating the cosine term with a finite Taylor series with only one control point $t_0$ is highly inefficient if not infeasible over multiple periods, requiring exorbitantly many degrees of freedom. To preserve the (quasi-)periodicity of $s(t)$, the function is nonuniformly sampled at 
$$
\mathcal{D} =  \left\{t_k: s(t_k) = \frac{2\pi}{K} k, k \in \mathbb{I} \right\}.
$$
Here, we assume that $r_M(t)\phi_o(t)$ is a monotonically increasing function (i.e., $\dot{\phi}_o(t)=\omega_o(t)>0$, $r_M(t)>0$, and $|\phi_o(t)\dot{r}_M(t)|<\omega_o(t)r_M(t)$). Furthermore, the sampling spacing is reduced by defining $K-1$ additional uniformly spaced sampling points between $t_i$ and $t_{i+1}$, resulting in the set of time points: 
$$
\mathcal{D} = \left\{t_i + \frac{k-iK}{K}t_{i+1}: k \in \mathbb{I}, i = \left\lfloor{\frac{k}{K}}\right\rfloor \right\}
$$

There are two approaches: (1) weighted averaging of finite Taylor expansion or (2) (B-spline) interpolating the approximates evalauted at $t_i \in \mathcal{D}$.
$$
\phi_j = \frac{t_j-t_i}{t_{i+1}-t_i}\hat{\phi}^{(N)}_{FM}(t_j; t_i) + \frac{t_{i+1}-t_j}{t_{i+1}-t_i}\hat{\phi}^{(N)}_{FM}(t_j; t_{i+1}),
$$
where $\hat{\phi}^{(N)}_{FM}(t;t_0)$ is the $N$th degree Taylor expansion of $\phi_{FM}(t)$ evaluated at $t_0$ and $t_i$ is the closest earlier sample point in $\mathcal{D}$ to $t_j$.

and 
$$
\mathcal{R} = \left\{\hat{\phi}^{(N)}_{FM}(t_i;t_i): t \in \mathcal{D} \right\},
$$
where $\hat{\phi}^{(N)}_{FM}(t;t_0)$ is the $N$th degree Taylor expansion of $\phi_{FM}(t)$ evaluated at $t_0$.



# Derivative of exponential function (including complex sinusoid)

$$
\begin{aligned}
\frac{d^n e^{x(t)}}{dt^n} &= e^{x(t)} \sum_{k=0}^n \frac{1}{k!} \sum_{j=0}^k (-1)^j {k\choose j} \left[\frac{d^n x(t)}{dt^n}\right]^{-j + k} x(t)^j\\
&= e^{x(t)} \sum_{k=0}^n \sum_{j=0}^k \frac{(-1)^j}{j!(k-j)!} \left[\frac{d^n x(t)}{dt^n}\right]^{k-j} x(t)^j\\
\end{aligned}
$$ for ($n$ element $Z$ and $n>=0$)
