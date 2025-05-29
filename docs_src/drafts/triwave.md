## 2nd order asymmetric waveform

Asymmetric triangle wave
$$
\begin{align*}
a_0 &= 0\\
a_n &= 0\\
b_n &= - \frac{2(-1)^nm^2}{n^2(m-1)\pi^2} \sin \frac{n(m-1)\pi}{m}\\
\end{align*}
$$

$$
f(x) = \frac{-2m^2}{(m-1)\pi^2} \sum_{n=1}^{\infty} \frac{(-1)^n}{n^2} \sin \frac{n(m-1)\pi}{m} \sin n \omega x\\
$$

$$
\begin{align*}
f(x-\tau) &= \sum_{n=1}^{\infty} b_n \sin n \omega (x-\tau)\\
&= \sum_{n=1}^{\infty} b_n (\sin n \omega x \cos n \omega \tau - \cos n \omega x \sin n \omega \tau)\\
&= \sum_{n=1}^{\infty} (- b_n \sin n\omega \tau) \cos n \omega x + (b_n \cos n \omega \tau) \sin n \omega x\\
\end{align*}
$$
