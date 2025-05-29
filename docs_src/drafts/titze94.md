$$P_L = R_g(t) u(t) + I \dot{u}(t)$$

- $P_L$ - the lung pressure
- $R_g(t)$ - a nonlinear kinetic resitance
- $u(t)$ - the glottal flow
- $I$ - a lumped inertance of the vocal tract air column
- $\dot{u}(t)$ - the time derivative of the flow

$$R_g(t) = \frac{k_t \rho |u(t)|}{2 a(t)}$$

- $k_t$ - a transglottal pressure coefficient
- $\rho$ - the density of air
- $a(t)$ - the minimum cross-sectional area in the glottis

Combine and solve for $\dot{u}$:

$$
\begin{align*}
P_L &= \frac{k_t \rho |u(t)| u(t)}{2 a(t)} + I \dot{u}(t) \\
1 &= \frac{k_t \rho}{2 P_L}\frac{|u(t)| u(t)}{a(t)} + \frac{I}{P_L} \dot{u}(t)
\end{align*}
$$

Define the no-load ($I=0$) particle velocity $v_0$ in the glottis:

$$v_0^2 = \frac{2P_L}{k_t\rho}$$

$$
\gamma = \frac{I}{2P_L}
$$

Substitute $v_0^2$ and $\gamma$ into the ODE:

$$
\begin{align*}
\frac{2P_L}{k_t\rho} &= \frac{|u(t)| u(t)}{a(t)} + \frac{2 I}{k_t\rho} \dot{u}(t)\\
v_0^2 &= \frac{|u(t)| u(t)}{a(t)} + 2 \gamma \dot{u}(t)\\
a(t) v_0^2 &= |u(t)| u(t) + 2 \gamma a(t) \dot{u}(t)\\
\end{align*}
$$

Let $a \dot{u} = \dot{w}$, $a u = w$:

$$
\begin{align*}
a(t) v_0^2 &= |u(t)| u(t) + 2 \gamma \dot{w}\\
\end{align*}
$$

Define the inertial load factor $\delta = \gamma a v_0$

into the ODE:

$$
\begin{align*}
1 &= \frac{|u(t)| u(t)}{a(t) v_0^2} + 2 \gamma \dot{u}(t)
\end{align*}
$$

Let $u = av_0 w$, $\dot{u} = \dot{a}v_0 w + av_0 \dot{w}$

$$
\begin{align*}
\frac{|a v_0 w| a v_0 w}{a v_0^2} + 2 \gamma (\dot{a}v_0 w + av_0 \dot{w}) &= 1\\
|w|w a   + 2 \gamma\dot{a}v_0 w   + 2 \gamma av_0 \dot{w} &= 1\\
\end{align*}
$$

Convert to the `scipy.integrate.ode` compatible form

$$
 \dot{w} = \frac{1 - |w|w a  - 2 \gamma\dot{a}v_0 w} {2 \gamma a v_0}
$$

$$
\dot{u} = \frac{1}{2 \gamma}\left[1 - \frac{|u| u}{a v_0^2}\right]
$$

Jacobian of the right-hand side:

$$
J(u) = \begin{cases}
    -\frac{u}{\gamma a v_0^2} & \text{if } u \ge 0 \\
    \frac{u}{\gamma a v_0^2} & \text{otherwise.}
\end{cases}
$$

The minimum area function:

$$
a(t) = \max{\left(a_\text{min}, \sin^\beta \frac{\pi t}{Q_0 T_0} \right)}
$$

where

- $a_\text{min}$ - 0 if complete closure or >0 for a gap (breathiness)
- $Q_0$ - open quotient
- $T_0$ - fundamental period
- $\beta = Q_0 + 1.0$ - measure of the softness of tonset and offset of each pulse

## Difference equation approximation

$$
\frac{u_n - u_{n-1}}{\Delta_t} = \frac{1}{2 \gamma}\left[1 - \frac{|u_n| u_n}{a_n v_0^2}\right]
$$

$$\delta_n= \frac{\gamma a_n v_0^2}{\Delta_t}$$

$$
2 \delta_n (u_n - u_{n-1}) = a_n v_0^2 - |u_n| u_n
$$

$$
u_n^2 + 2 \delta_n u_n - a_n v_0^2 - 2 \delta_n u_{n-1} =  0
$$

$$
u_n = \sqrt{\delta_n^2 + a_n v_0^2 + 2 \delta_n u_{n-1}} - \delta_n
$$
