## Telegrapher's Equations

$$
\begin{align}
\frac{dP}{dx}&=ZU\\  
\frac{dU}{dx}&=YP\\  
\end{align}
$$

Discretize space

$$
\begin{align}
  P_2-P_1 = \Delta x Z U\\
  U_2-U_1 = \Delta x Y P\\
\end{align}
$$

### Lossless Tract:

Lossless tract has $Z=s \rho/A$ and $Y=sA/\rho c^2$.

$$
\begin{align}
  P_2-P_1 = \Delta x s \frac{\rho}{A} U\\
  U_2-U_1 = \Delta x s \frac{A}{\rho c^2} P\\
\end{align}
$$

The pressure and flow are related by $P=Z_0U$ with the characteristic impedance $Z_0=\sqrt{Z/Y}$. The impedance for lossless tract is 

$$
Z_0=\sqrt{\frac{s\rho}{A}\frac{\rho c^2}{s A}}=\frac{\rho c}{A}
$$


$$
\begin{align}
  P_2-P_1 &= \frac{\Delta x}{c} s P\\
  U_2-U_1 &= \frac{\Delta x}{c} s U\\
\end{align}
$$


$$
\begin{align}
  P_{2,n}-P_{1,n} &= \frac{\Delta x}{\Delta t c} \left(P_{1,n+1}-P_{1,n}\right)\\
  U_{2,n}-U_{1,n} &= \frac{\Delta x}{\Delta t c} \left(U_{1,n+1}-U_{1,n}\right)\\
\end{align}
$$

If $\Delta x/\Delta t = c$ then,

$$
\begin{align}
P_{2,n} &= P_{1,n+1}\\
U_{2,n} &= U_{1,n+1}\\  
\end{align}
$$

### Lossy Tract:

$$
Z=R+sL
$$


$$
\begin{align}
  P_2-P_1 = \Delta x \left(R + s \frac{\rho}{A}\right) U_1\\
  U_2-U_1 = \Delta x s \frac{A}{\rho c^2} P_1\\
\end{align}
$$



$$
\begin{aligned}
Z_0 &= \sqrt{\frac{Z}{Y}}=\sqrt{\frac{R+s\rho/A}{sA/\rho c^2}}\\
  &=\frac{\rho c}{A} \sqrt{\frac{R A}{s \rho} + 1}\\  
\end{aligned}
$$



VT has significant shunt inductance compared to electrical transmission line, requiring a "1D Yee leapfrog arrangement" to run its simulation using finite-difference time-domain simulation. This method appears to be an equivalent to the Liljencrants' model.

To simulate the modified lossy telegrapher’s equations using the Finite-Difference Time-Domain (FDTD) method, you can use a 1D Yee leapfrog arrangement. The voltage, series current, and shunt inductor current are staggered in space and time to achieve a numerically stable, explicit update loop.

## 1. The Discretization Scheme

The transmission line is divided into spatial steps $\Delta z$ (indexed by $k$) and time steps $\Delta t$ (indexed by $n$).

* Voltage ($V_k^n$): Positioned at integer nodes $k$ and integer time steps $n$.
* Series Current ($I_{k+1/2}^{n+1/2}$): Positioned at half-integer spatial nodes and half-integer time steps.
* Shunt Inductor Current ($I_{L_{sh}, k}^{n+1/2}$): Positioned at integer spatial nodes $k$ (matching $V$) but half-integer time steps.

---

## 2. The FDTD Update Equations

At each time step, you solve for the future values explicitly using central-difference approximations. [1]

## Step 1: Update the Shunt Inductor Current

The current through the extra shunt inductor accumulates based on the local voltage:

$$
I_{L_{sh}, k}^{n+1/2} = I_{L_{sh}, k}^{n-1/2} + \frac{\Delta t}{L_{sh}} V_k^n
$$

## Step 2: Update the Series Current

The series current updates based on the spatial gradient of the voltage:

$$
I_{k+1/2}^{n+1/2} = \left( \frac{2L - R\Delta t}{2L + R\Delta t} \right) I_{k+1/2}^{n-1/2} - \left( \frac{2\Delta t}{(2L + R\Delta t)\Delta z} \right) (V_{k+1}^n - V_k^n)
$$

## Step 3: Update the Voltage

The voltage updates based on the spatial gradient of the series current and the shunt leakage:

$$
V_k^{n+1} = \left( \frac{2C - G\Delta t}{2C + G\Delta t} \right) V_k^n - \left( \frac{2\Delta t}{(2C + G\Delta t)\Delta z} \right) \left( I_{k+1/2}^{n+1/2} - I_{k-1/2}^{n+1/2} \right) - \left( \frac{2\Delta t}{2C + G\Delta t} \right) I_{L_{sh}, k}^{n+1/2}
$$

---

## 3. Simulation Constraints & Stability

* Stability Limit: The time step must satisfy the Courant-Friedrichs-Lewy (CFL) condition:

$$
\Delta t \le \frac{\Delta z}{v_{max}} = \Delta z \sqrt{LC}
$$

* Shunt Influence: The extra shunt inductance $L_{sh}$ dictates the low-frequency cutoff but does not alter the high-frequency speed limit ($1/\sqrt{LC}$); hence, it does not change the CFL limit. [2]
