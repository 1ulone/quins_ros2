## Timing
so, scene.xml can sets its own timestep with `<option timestep>` but since scene.xml didnt set one. Mujoco will default to 0.002 s or about 500Hz of physics step

step are process of thing updating, so physics step of 0.002s means for every 0.002 second it'll update the physics process / calculation.

and then from the physics step there. every system follows with varied timestep:

- physisc (mj_step) = 500 Hz
- TaskController + FOSMC + WBC = 500 Hz
- RLTuner.Observe = 500 Hz
- RLTuner.step (updating gain) = 50 Hz
- GaitLogic.loop_step = 50 Hz
- Console print = 2 Hz
- GUI Thread

## How the data flows :
1. Every 10 steps -> GaitLogic.loop_step, Decides : state, planner, start time, target yaw, etc.
2. TaskController.Compute -> asks GaitLogic.wbc_target(t)for stance, flags, foot, offset/vel/acc, Fz guess builds, e,e_dot for 18 task rows (body xyz, body rotation, 4 feet xyz). Then FOSMC: e, e_dot commanded task accelerations.
3. WholeBodyController.solve (QP) -> finds a, Fc, tau that best realize those accelerations subject to rigid-body dynamics, friction cones, torque limits
4. data.ctrl -> Motor * tau --> mj_step (Mujoco)
5. RL Tuner reads s and tau, retunes FOSMC gains every 20 ms

## How the control system works.
On GaitPlanner calculate foot paths then KinematicsLogic Calculate FK/ IK/ Jacobian
$$
c = (t\,f + o_{leg}) \bmod 1, \qquad
\begin{cases} \text{stance}, & s = c / d \quad \text{if } c < d \\ \text{swing}, & s = (c - d)/(1 - d) \quad \text{otherwise} \end{cases}
$$
with 
- $c$ = Cycle (one foot cycle swing and stance)
- $f$ = gait frequency
- $d$ = duty factor
- $s$ = Progress of current stance

The code then written as :
```python
cycle = (t * self.p.freq + self.p.offsets[leg]) % 1.0
```

For the swing foot height with z as the axis up and down it uses polynominal that peaks at  the 
step height $h$ when $s\ =\ 0.5$ 
$$z(s) = 64\ h\ s^3(1\ -\ s)^3$$
The forward swing motion then is a mininmal jerk quintic, $10s^3\ -\ 15s^4\ +\ 6s^5$ 
The code then written as :
```python
return 64.0 * self.p.step_h * (s**3) * ((1.0 - s)**3)
```

Foot velocity and acceleration are numerical derivatives of the path that are computed in `foot_vel()` and `foot()` 

KinematicsLogic which hardware would probably uses.
FK is the product of the joint transform from the URDF, it is not written down mathemathically in code, but the library **Pinocchio** translate it with the equation :
$$^BT_F(q)=T_{BH}\ R_x(q_1)\ T_{HT}\ R_y(q_2)\ T_{TS}\ R_y(q_3)\ T_{SF}$$
with 
- $B$ = Body (base)
- $H$ = Hip
- $T$ = Thigh
- $S$ = Shank
- $F$ = Foot
Which make the value became : 
- $T_{BH}$ = Body center to the hip joint (where LF_HAA joint sits on the body)
- $T_{HT}$ = hip joint to the thigh joint (HAA to HFE)
- $T_{TS}$ = thigh joint to the knee (HFE to KFE)
- $T_{SF}$ = knee to the foot tip

Each of those $T_{xy}$ contain a constant 4x4 transform, a translation plus any fixed rotation (which usually is 0) : 
$$T = \begin{bmatrix} R_0 & d \\ 0 & 1 \end{bmatrix}$$
with $d$ is the offset vector in meters and $R_0$ is the fixed rotation.
The pinocchio then wrote equation as :
```python
pin.forwardKinematics(self.model, self.data, q)
pin.updateFramePlacements(self.model, self.data)
return self.data.oMf[self.foot_ids[leg_id]].homogeneous
```
The jacobian $\dot{q} = J(q)\ \dot{q}$  comes from `get_jacobian()` function, which keeps the 3x3 linear part for one leg.
IK then uses the damped least squares, with $\lambda=10^{-6},\ \alpha=0.5$ , with tolerance of $10^{-4}$ m and at most 50 iterations:
$$\triangle q=-J^T(JJ^T+\lambda I)^{-1}(p(q)-p_{target}),\\\ q \leftarrow q+\alpha \ \triangle q$$
written in code as :
```python
v_leg = -J_leg.T @ np.linalg.solve(J_leg @ J_leg.T + damp * np.eye(3), err)
q[idx_q] += v_leg * alpha
```
Then on `GaitLogic.gait_process()` it'll take the planner's foot target, runs IK to get joint angles, then on those angle we can get the foot velocity to joint velocity with the same damped inverse of $\dot{q}=J^T(JJ^{T}+\lambda I)^{-1}\dot{p}$ , clipped to $\pm40$  rad/s.

after we got the leg velocity (joint velocity), we then calculate or get the contact detection. that used the estimation logic. it detects when a leg / foot touches a ground when $f_z > 10\ N$  and stops touching when $f_z < 5\ N$, with a 3-sample debounce. which is a small state machine with three "state" SWING, STANCE, and SEARCH. those state handles feet that land either early or late.

A fake ground plane then is theoreatically built / stored inside the foot 4x3 matrix through the function `touchdown(self, i, point)` with the point as the foot center minus the foot radius so its literally the center surface of the foot . with the plane (ground) that can be written as 
$$z=a_0 + a_1x+a_2y$$
With 3 value there are unknown and 4 point that we have. which more data that we have than the unknonw one. so no plalne passes through all four (points) exactly. so we use/find the least squares to pick the $a$ value that minimize the total squared height error over the 4 points, with the equation.
$$\min_a\; \sum^4_{i=1}(z_i-(a_0+a_1x_i+a_2y_i))^2$$
that equation is written as one line in the function `_fit()` 
```python
A = np.column_stack([np_ones(len(self.pts)), self.pts[:, 0:2]])
self.raw = np.linalg.lstsq(A, self.pts[:, 2], rcond=None)[0]
```
the condition then became : 
- all four z values are equal so $a_1 = a_2 = 0$ and so on, means that **Floor is Flat**
- the four z valus is different as in the $a1 = x$ and $a4 = 0$ means that probably the current **Floor is a Ramp**

Now for the control system. there are 18 task to execute/control. with containing :
- body position (3) $(p_x, p_y, p_z)$
- body rotation (3) $(r_x, r_y, r_z)$
- four feet (3 each) written like:
$$\begin{bmatrix} 
f_11 & f_12 & f_13\\
f_21 & f_22 & f_33\\
f_31 & f_32 & f_33\\
f_41 & f_42 & f_43\\
\end{bmatrix}$$
To compute each errors for the 18 task are as follow :
- Body (x, y) : $e=p_{xy}-p^d_{xy}$ , The target moves at the commanded velocity of  $v_{end}=\frac{\ell \ \cdot \ f}{d}$ , with $\ell$ as step length.
- Body Height : $e_z = z - (g(x,y) - z_{target})$ , measured relative to the estimated ground $g$.
- Body Rotation : $e_R = log(R\ R_d)$ ,computed by `so3_log()`
- Feet : follow the planner path. the touchdown point is corrected with the *Raibert rule*, $P^d_{xy} += k_{foot}s(v - v_{cmd})$ 

The PD law maps errors to desired accelerations, with $K_d = 2\sqrt{K_p}$ :
$$a_{cmd} = a_{ff} - K_pe - K_d\dot{e}$$
from the paper referenced FOSMC (Farid 2023, eq 31) it uses this sliding surface :
$$s = \dot{e}+K_{e1}sig(e)^\lambda + K_{e2} \ D^{\alpha-1}e$$
in here $sig(e)^\lambda=sign(e)\  |e|^\lambda$ . The Control law is :
$$a_{cmd} = a_{ff} - \lambda K_{e1}(|e| + \epsilon)^{\lambda-1}\dot{e}-K_{e2}D^{\alpha}e-K_ss-K_rsat(s)$$
in the code it is then written as 
```python
sig = np.sign(e) * np.abs(e) ** self.lam
s = e_dot + self.Ke1 * sig + self.Ke2 * d_s
```

The fractional derivative is computed with the Grunwald-Letnikov sum over the last N = 100 samples, with $\beta = \alpha - 1$ :
$$D^{\beta} x(t) \approx \frac{1}{h^{\beta}} \sum_{j=0}^{N-1} w_j\, x(t - jh), \qquad w_0 = 1, \quad w_j = w_{j-1}\left(1 - \frac{\beta + 1}{j}\right)$$
this is implemented in `gl_weights()`. it computes $D^\alpha e$ as $D^{\alpha-1} \dot{e}$ , which avoids differentiating the error history.

The Dynamic Model is using adam.casadi library. which evaluate the urdf with the CasADi functions. using the MIXED velocity representation. the code is written like :
```python
M = self.kindyn.mass_matrix(w_H_b, q)
h = self.kindyn.bias_force(w_H_b, q, v_b, v_j)
```
It also builds $J_c$ the term $\dot{j}_cv$, adn the foot positions $p_f$ , and packs everything into one function, `wbc_dyn`.

The decision variable are $x = [a, F_c, \tau]$ : generalized accelerations (18), contact forces (12) and joint torques (12). The cost is :
$$
\min_x W_{lin}||a_{0:3} - a^d_{lin}||^2 + W_{ang}||a_{3:6}-a^d_{ang}||^2
\newline + \sum_i(1-s_i)W_{sw}||J_ia + \dot{J}_iv - a^d_{foot,i}||^2 + \sum_i s_i W_c || J_ia + \dot{J}_iv + k_d J_iv||^2
\newline + W_{post} || \ddot{q}-\ddot{q}_{ref} ||^2 + W_{\tau}||\tau||^2 + W_f + ||F_c - F_{ref}||^2
$$
The swing-foot term is active when $s_i = 0$ and the soft no-slip term when $s_i = 1$.
the QP is subject to :
- Equation of motion, with joint damping D matched to Mujoco :
$$ M(q)a + h(q, v) = \begin{bmatrix} 0 \\ \tau - D\dot{q} \end{bmatrix} + J^T_c F_c $$
- Torque limits : $| \tau |<= \tau_{max}$
- Joint limits as acceleration bounds (from Del prete 2018 ? ):
  $$ \ddot{q}_{hi} = min(\frac{2(q_{hi}-q-\dot{q}T)}{T^2}, \frac{\sqrt{2 \ddot{q}_{max}(q_{hi}-q)}-\dot{q}}{\triangle t}) $$
- Friction pyramid, a linearized cone : $|f_x| \le \mu f_z$, $|f_y| \le \mu f_z$, and $0 \le f_z \le f_{z,max} \cdot s_i$.
the code is written as :
```python
net = M @ a + h - Jc.T @ Fc
opti.subject_to(net[0:6] == 0)
opti.subject_to(net[6:] + self.joint_damping * v[6:] - tau == 0)
```
CasAdi then build that as a QP (Opti ('conic')) and solves it with qpOASES for each tick.

Mujoco will integrates its own equation of motion and computes ist own contact forces. The controller reads only the state and the measured foot forces (`mj_contactForce`). The `assert` on `dof_damping` ensures the WBC's D matches the simulator.

## Functions 
over generalization of the whole Archicteture : 
- Scene.xml function as the Mujoco "model" of the robot and the environment, it sets the initial mass, friction, etc of the robot.
- MujocoSim function as the main .py that runs and calls the whole archicteture.
- GaitLogic function as the controller for time and intent, which state on which gait phase (using gait clock)
- GaitPlanner functino as the "planner" as per the name, it plans the trajectory using geometry (in this case a 5th order polynomial for the swing trajectory).
- TaskController functions as the error reader, it reads and give detailed output about where the body and feet should be.
- KinematicsLogic function as the per leg fk and ik using pinocchio library, it is used to compute the nominal foot positions and the lowered walking and idle poses.
- FOSMCLogic functions as the corrector. it reads the TaskController evaluation on where the feet should be, then FOSMC will try to correct those error with the pd gain scaling how hard the FOSMC would correct them
- WBCLogic function as the Ruleset. before sending torque and what not to the actual robot, the wbc constraint them. it limits any value that violates friction, torque, limits, or dynamics.
- Critic is a RBF Network that outputs shifts to the FOSMC gains. it learns from the temporal-difference error that TaskController suppposedly read.
- Actor is also a RBF Network that outputs shifts to the FOSMC gains. it learns from the critic judgement to wheteher change an action for better or worse.
- MainTuner connects the tuner to the controller (wbc fosmc and such) builds the critic input and also run both actor and critic. and applied their result to the fosmc
- PreTrain offline training which it'll run 20 randomized headless episode that compares the baseline againts the learned gaits.
