# Klatt Pitch Flutter

Phase-(instantaneous-)frequency relationship:

$$
\phi = \int_{0}^t \omega_o(\tau)d\tau + \phi_0
$$

Generalized equation for the flutter effect in Klatt (1990)

$$\Delta_f (t) = c f_o(t) \sum_i \sin \omega_i t + \phi_i $$

## If $\Delta_f (t)$ is a phase fluctuation function

Its implementations (Praat, eSpeak) seem to imply that $\Delta f(t)$ is actually modifies the phase (not instantaneous frequency), i.e.,

$$
\begin{align}
\phi &= \int_{0}^t \omega_o(\tau)d\tau + 2 \pi \Delta_f(t) t + \phi_0\\
     &= \int_{0}^t \omega_o(\tau) d\tau  + \phi_\Delta (t)t + \phi_0
\end{align}
$$

Here, $\phi_\Delta (t) = 2\pi \Delta f(t)$. Bring the flutter term inside the integral:

$$
\phi = \int_{0}^t \left[ \omega_o(\tau) + \phi_\Delta(\tau) + \dot \phi_\Delta(\tau) t  \right] d\tau + \phi_0
$$

where

$$
\begin{align}
\frac{\dot \phi_\Delta (t)}{2\pi} &= \frac{\partial \Delta_f}{\partial t}\\
&= c \left[ \frac{\partial f_o}{\partial t} \sum_i \sin \omega_i \tau  + f_o(t) \sum_i \omega_i \cos \omega_i t\right]
\end{align}
$$

This solution, however, unboundedly increases the instantaneous frequency due to the $\dot \phi_\Delta(\tau) \tau$ term.

## If $\Delta_f (t)$ is an instantaneous frequency fluctuation

The flutter definition does imply it to be applied to the instantaneous frequency, instead.

$$
\begin{align}
\tilde \omega_o(t) &= \omega_o(t) + 2 \pi \Delta_f(t)\\
            &= \omega_o(t) + 2 \pi c f_o(t) \sum_i \sin \omega_i t + \phi_i\\
            &= \omega_o(t) \left[1 + c \sum_i \sin \omega_i t + \phi_i \right]\\
            &\triangleq \omega_o(t) g(t)
\end{align}
$$

The resulting phase function is then

$$
\begin{align}
\phi &= \int_{0}^t \omega_o(\tau) g(t) d\tau + \phi_0\\
&= \omega_o(\tau) \int_0^t g(\tau) d\tau - \int_0^t \dot \omega_o(\tau) \left[\int g(\tilde t) d\tilde t\right]_{\tilde t = \tau} d\tau + \phi_0 \\
\end{align}
$$

The second line results from the integration by parts.

The integral of $g(t)$ has a closed-form solution:

$$
\begin{align}
\int g(t) dt &= \int \left[1 + c \sum_i \sin \left(\omega_i t + \phi_i\right)\right] dt\\
&= t - \frac{c}{\omega_i} \sum_i \cos \left(\omega_i t + \phi_i\right) dt\\
\end{align}
$$

Substituting this expression into $\phi$ yields

$$
\begin{align}
\phi &= \omega_o(\tau) t - \frac{c \omega_o(\tau)}{\omega_i} \sum_i \cos \left(\omega_i t + \phi_i\right) - \int_0^t \dot \omega_o(\tau) \left[\tau - \frac{c}{\omega_i} \sum_i \cos \left(\omega_i \tau + \phi_i\right) \right] d\tau + \phi_0 \\
\end{align}
$$

If $\omega_o(t)$ is constant, the integral term vanishes, yielding

$$
\phi = \omega_o(\tau) t - c \omega_o(\tau) \sum_i \frac{\cos \left(\omega_i t + \phi_i\right)}{\omega_i}  + \phi_0
$$

If $\omega_o(t) = a_0 + a_1 t$ (a linear chirp), $\dot \omega_o(t) = a_1$, then

$$
\begin{align}
\int_0^t \dot \omega_o(\tau) \left[\tau - \frac{c}{\omega_i} \sum_i \cos \left(\omega_i \tau + \phi_i\right) \right] d\tau &= \int_0^t a_1 \left[\tau - c \sum_i \frac{\cos \left(\omega_i \tau + \phi_i\right)}{\omega_i} \right] d\tau \\
&= \frac{a_1}{2} t^2 - a_1 c \sum_i \frac{\sin \left(\omega_i t + \phi_i\right)}{\omega_i^2}\\
\end{align}
$$


If $\omega_o(t) = a_0 + a_1t + a_2 t^2$ (a quadratic chirp), $\dot \omega_o(t) = a_1 + a_2t$, then

$$
\begin{align}
\int_0^t \dot \omega_o(\tau) & \left[\tau - \frac{c}{\omega_i} \sum_i \cos \left(\omega_i \tau + \phi_i\right) \right] d\tau = \int_0^t (a_1 + a_2\tau) \left[\tau - c \sum_i \frac{\cos \left(\omega_i \tau + \phi_i\right)}{\omega_i} \right] d\tau \\
&= \frac{a_1}{2} t^2 - a_1 c \sum_i \frac{\sin \left(\omega_i \tau + \phi_i\right)}{\omega_i^2}  + \frac{a_2}{3} t^3 - \int_0^t a_2 c \tau  \sum_i \frac{\cos \left(\omega_i \tau + \phi_i\right)}{\omega_i} d\tau\\
\end{align}
$$
