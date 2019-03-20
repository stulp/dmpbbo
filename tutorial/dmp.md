Dynamical Movement Primitives
===============

The core idea behind dynamical movement primitives (DMPs) is to represent movement primitives as a combination of dynamical systems (please read <a class="el" href="page_dyn_sys.html">Dynamical Systems</a>, if you haven't already done so). The state variables of the main dynamical system ![alt text](formulae/form_0.png "$ [\mathbf{y~\dot{y}~\ddot{y}} ]$")  then represent trajectories for controlling, for instance, the 7 joints of a robot arm, or its 3D end-effector position. The attractor state is the end-point or *goal* of the movement.

The key advantage of DMPs is that they inherit the nice properties from linear dynamical systems (guaranteed convergence towards the attractor, robustness to perturbations, independence of time, etc) whilst allowing arbitrary (smooth) motions to be represented by adding a non-linear forcing term. This forcing term is often learned from demonstration, and subsequently improved through reinforcement learning.

DMPs were introduced in <a class="el" href="citelist.html#CITEREF_ijspeert02movement">[3]</a>, but in this section we follow largely the notation and description in <a class="el" href="citelist.html#CITEREF_ijspeert13dynamical">[4]</a>, but at a slower pace.

*Historical remark*. The term "dynamicAL movement primitives" is now preferred over "dynamic movement primitives". The newer term makes the relation to dynamicAL systems more clear, and avoids confusion about whether the output of "dynamical movement primitives" is in kinematic or dynamic space (it is usually in kinematic space).

*Remark*. This documentation and code focusses only on discrete movement primitives. For rythmic movement primitives, we refer to <a class="el" href="citelist.html#CITEREF_ijspeert13dynamical">[4]</a>.


<a name="sec_core"></a>

Basic Point-to-Point Movements: A Critically Damped Spring-Damper System
---------------

At the heart of the DMP lies a spring-damper system, as described in <a href="page_dyn_sys.html#dyn_sys_spring_damper">Spring-Damper Systems</a>. In DMP papers, the notation of the spring-damper system is usually a bit different: 


![alt text](formulae/form_1.png "\begin{eqnarray*} m\ddot{y} =&amp; -ky -c\dot{y} &amp; \mbox{spring-damper system, ``traditional notation''} \\ m\ddot{y} =&amp; c(-\frac{k}{c}y - \dot{y})\\ \tau\ddot{y} =&amp; \alpha(-\beta y - \dot{y}) &amp; \mbox{with } \alpha=c,~~\beta = \frac{k}{c},~~m=\tau\\ \tau\ddot{y} =&amp; \alpha(-\beta (y-y^g) - \dot{y})&amp; \mbox{with attractor } y^g\\ \tau\ddot{y} =&amp; \alpha(\beta (y^g-y) - \dot{y})&amp; \mbox{typical DMP notation for spring-damper system}\\ \end{eqnarray*}") 


In the last two steps, we change the attractor state from 0 to ![alt text](formulae/form_2.png "$y^g$") , where ![alt text](formulae/form_2.png "$y^g$")  is the goal of the movement.

To avoid overshooting or slow convergence towards ![alt text](formulae/form_2.png "$y^g$") , we prefer to have a <em>critically</em> <em>damped</em> spring-damper system for the DMP. For such systems ![alt text](formulae/form_3.png "$c = 2\sqrt{mk}$")  must hold, see <a class="el" href="page_dyn_sys.html#dyn_sys_critical_damping">Critical Damping</a>. In our notation this becomes ![alt text](formulae/form_4.png "$\alpha = 2\sqrt{\alpha\beta}$") , which leads to ![alt text](formulae/form_5.png "$\beta = \alpha/4$") . This determines the value of ![alt text](formulae/form_6.png "$\beta$")  for a given value of ![alt text](formulae/form_7.png "$\alpha$")  in DMPs. The influence of ![alt text](formulae/form_7.png "$\alpha$")  is illustrated in the first figure <a class="el" href="page_dyn_sys.html">here</a>.

Rewriting the second order dynamical system as a first order system (see <a class="el" href="page_dyn_sys.html#dyn_sys_rewrite_second_first">Rewriting one 2nd Order Systems as two 1st Order Systems</a>) with expanded state ![alt text](formulae/form_107.png "$ \mathbf{x}= [z~y]$")  yields:



![alt text](formulae/form_102.png "\begin{eqnarray*} \mathbf{\dot{x}} = \left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \end{array} \right] = \left[ \begin{array}{l} (\alpha (\beta({y}^{g}-{y})-{z}))/\tau \\ {z}/\tau \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {0} \\ {y}^g \end{array} \right] \end{eqnarray*}") 


Please note that in the implementation, the state is implemented as ![alt text](formulae/form_10.png "$ [y~z]$") . The order is inconsequential, but we use the notation above ( ![alt text](formulae/form_11.png "$[z~y]$") ) throughout the rest of this tutorial section, for consistency with the DMP literature.


<a name="sec_forcing"></a>

Arbitrary Smooth Movements: the Forcing Term
---------------

The representation described in the previous section has some nice properties in terms of <a class="el" href="page_dyn_sys.html#sec_dyn_sys_convergence">Convergence towards the Attractor</a> , <a class="el" href="page_dyn_sys.html#sec_dyn_sys_perturbations">Robustness to Perturbations</a> , and <a class="el" href="page_dyn_sys.html#sec_dyn_sys_autonomy">Autonomy</a>, but it can only represent very simple movements. To achieve more complex movements, we add a time-dependent forcing term to the spring-damper system. The spring-damper systems and forcing term are together known as a <em>transformation</em> <em>system</em>.



![alt text](formulae/form_103.png "\begin{eqnarray*} \mathbf{\dot{x}} = \left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \end{array} \right] = \left[ \begin{array}{l} (\alpha (\beta({y}^{g}-{y})-{z}) + f(t))/\tau \\ {z}/\tau \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {?} \\ {y}^g \end{array} \right] \end{eqnarray*}") 


The forcing term is an open loop controller, i.e. it depends only on time. By modifying the acceleration profile of the movement with a forcing term, arbitrary smooth movements can be achieved. The function ![alt text](formulae/form_13.png "$ f(t)$")  is usually a function approximator, such as locally weighted regression (LWR) or locally weighted projection regression (LWPR), see <a class="el" href="page_func_approx.html">Function Approximation</a>. The graph below shows an example of a forcing term implemented with LWR with random weights for the basis functions.


![alt text](images/dmp_forcing_terms-svg.png  "A non-linear forcing term enable more complex trajectories to be generated (these DMPs use a goal system and an exponential gating term).")


<a name="sec_forcing_convergence"></a>

Ensuring Convergence to 0 of the Forcing Term: the Gating System
---------------

Since we add a forcing term to the dynamical system, we can no longer guarantee that the part of the system repesenting ![alt text](formulae/form_79.png "$ y $")  will converge towards ![alt text](formulae/form_35.png "$ y^g $") ; perhaps the forcing term continually pushes it away ![alt text](formulae/form_35.png "$ y^g $")  (perhaps it doesn't, but the point is that we cannot <em>guarantee</em> that it <em>always</em> doesn't). That is why there is a question mark in the attractor state in the equation above.

To guarantee that the movement will always converge towards the attractor ![alt text](formulae/form_35.png "$ y^g $") , we need to ensure that the forcing term decreases to 0 towards the end of the movement. To do so, a gating term is added, which is 1 at the beginning of the movement, and 0 at the end. This gating term itself is determined by, of course, a dynamical system. In <a class="el" href="citelist.html#CITEREF_ijspeert02movement">[3]</a>, it was suggested to use an exponential system. We add this extra system to our dynamical system by expanding the state as follows:



![alt text](formulae/form_104.png "\begin{eqnarray*} \mathbf{\dot{x}} = \left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({y}^{g}-{y})-{z}) + v\cdot f(t))/\tau \\ {z}/\tau \\ -\alpha_v v/\tau \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {0} \\ {y}^g \\ 0 \end{array} \right] \end{eqnarray*}") 



<a name="sec_forcing_autonomy"></a>

Ensuring Autonomy of the Forcing Term: the Phase System
---------------

By introducing the dependence of the forcing term ![alt text](formulae/form_13.png "$ f(t)$")  on time ![alt text](formulae/form_16.png "$ t $")  the overall system is no longer autonomous. To achieve independence of time, we therefore let ![alt text](formulae/form_17.png "$ f $")  be a function of the state of an (autonomous) dynamical system rather than of ![alt text](formulae/form_16.png "$ t $") . This system represents the <em>phase</em> of the movement. <a class="el" href="citelist.html#CITEREF_ijspeert02movement">[3]</a> suggested to use the same dynamical system for the gating and phase, and use the term <em>canonical</em> <em>system</em> to refer this joint gating/phase system. Thus the phase of the movement starts at 1, and converges to 0 towards the end of the movement, just like the gating system. The new formulation now is (the only difference is ![alt text](formulae/form_18.png "$ f(x)$")  instead of ![alt text](formulae/form_13.png "$ f(t)$") ):



![alt text](formulae/form_105.png "\begin{eqnarray*} \mathbf{\dot{x}} = \left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({y}^{g}-{y})-{z}) + v\cdot f(v))/\tau \\ {z}/\tau \\ -\alpha_v v/\tau \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {0} \\ {y}^g \\ 0 \end{array} \right] \end{eqnarray*}") 


Note that in most papers, the symbol for the state of the canonical system is ![alt text](formulae/form_74.png "$ x $") . Since this symbol is already reserved for the state of the complete DMP, we rather use ![alt text](formulae/form_98.png "$ v$") 


*Todo*: Discuss goal-dependent scaling, i.e. ![alt text](formulae/form_99.png "$ f(t)v(y^g-y_0) $")


<a name="sec_multidim_dmp"></a>

Multi-dimensional Dynamic Movement Primitives
---------------

Since DMPs usually have multi-dimensional states (e.g. one output ![alt text](formulae/form_21.png "$ {\mathbf{y}}_{d=1\dots D}$")  for each of the ![alt text](formulae/form_22.png "$ D $")  joints), it is more accurate to use bold fonts for the state variables (except the gating/phase system, because it is always 1D) so that they represent vectors:



![alt text](formulae/form_112.png "\begin{eqnarray*} \mathbf{\dot{x}} = \left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g}-\mathbf{y})-\mathbf{z}) + v\cdot \mathbf{f}(v))/\tau \\ \mathbf{z}/\tau \\ -\alpha_v v/\tau \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}_0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ 0 \end{array} \right] \end{eqnarray*}") 


So far, the graphs have shown 1-dimensional systems. To generate D-dimensional trajectories for, for instance, the 7 joints of an arm or the 3D position of its end-effector, we simply use D transformation systems. A key principle in DMPs is to use one and the same phase system for all of the transformation systems, to ensure that the output of the transformation systems are synchronized in time. The image below show the evolution of all the dynamical systems involved in integrating a multi-dimensional DMP.


![alt text](images/dmpplot_ijspeert2002movement-svg.png  "The various dynamical systems and forcing terms in multi-dimensional DMPs.")


<a name="sec_dmp_alternative"></a>

Alternative Systems for Gating, Phase and Goals
---------------

The DMP formulation presented so far follows <a class="el" href="citelist.html#CITEREF_ijspeert02movement">[3]</a>. Since then, several variations have been proposed, which have several advantages in practice. We now describe some of these variations.


<a name="sec_dmp_sigmoid_gating"></a>
### Gating: Sigmoid System

A disadvantage of using an exponential system as a gating term is that the gating decreases very quickly in the beginning. Thus, the output of the function approximator ![alt text](formulae/form_25.png "$ f(x) $")  needs to be very high towards the end of the movement if it is to have any effect at all. This leads to scaling issues when training the function approximator.

Therefore, sigmoid systems have more recently been proposed <a class="el" href="citelist.html#CITEREF_kulvicius12joining">[6]</a> as a gating system. This leads to the following DMP formulation (since the gating and phase system are no longer shared, we introduce a new state variable ![alt text](formulae/form_113.png "$ s $")  for the phase term):


![alt text](formulae/form_115.png "\begin{eqnarray*} \left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{s}} \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g}-\mathbf{y})-\mathbf{z}) + v\cdot f(s))/\tau \\ \mathbf{z}/\tau \\ -\alpha_s s/\tau \\ -\alpha_v v (1-v/v_{\mbox{\scriptsize max}}) \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}_0 \\ 1 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ 0 \\ 0 \end{array} \right] \end{eqnarray*}") 


where the term ![alt text](formulae/form_28.png "$ v_{\mbox{\scriptsize max}}$")  is determined by ![alt text](formulae/form_29.png "$\tau $") 


<a name="sec_dmp_phase"></a>
### Phase: Constant Velocity System

In practice, using an exponential phase system may complicate imitation learning of the function approximator ![alt text](formulae/form_17.png "$ f $") , because samples are not equidistantly spaced in time. Therefore, we introduce a dynamical system that mimics the properties of the phase system described in <a class="el" href="citelist.html#CITEREF_kulvicius12joining">[6]</a>, whilst allowing for a more natural integration in the DMP formulation, and thus our code base. This system starts at 0, and has a constant velocity of ![alt text](formulae/form_30.png "$1/\tau$") , which means the system reaches 1 when ![alt text](formulae/form_31.png "$t=\tau$") . When this point is reached, the velocity is set to 0.



![alt text](formulae/form_116.png "\begin{eqnarray*} \dot{s} =&amp; 1/\tau \mbox{~if~} s &lt; 1 &amp; \\ &amp; 0 \mbox{~if~} s&gt;1 \\ \end{eqnarray*}") 

This, in all honesty, is a bit of a hack, because it leads to a non-smooth acceleration profile. However, its properties as an input to the function approximator are so advantageous that we have designed it in this way (the implementation of this system is in the TimeSystem class).

![alt text](images/phase_systems-svg.png  "Exponential and constant velocity dynamical systems as the 1D phase for a dynamical movement primitive.")

With the constant velocity dynamical system the DMP formulation becomes:



![alt text](formulae/form_117.png "\begin{eqnarray*} \left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{s}} \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g}-\mathbf{y})-\mathbf{z}) + v\cdot f(s))/\tau \\ \mathbf{z}/\tau \\ 1/\tau \\ -\alpha_v v (1-v/v_{\mbox{\scriptsize max}}) \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}_0 \\ 0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ 1 \\ 0 \end{array} \right] \end{eqnarray*}") 



<a name="sec_delayed_goal"></a>
### Zero Initial Accelerations: the Delayed Goal System

Since the spring-damper system leads to high initial accelerations (see the graph to the right below), which is usually not desirable for robots, it was suggested to move the attractor of the system from the initial state ![alt text](formulae/form_34.png "$ y_0 $")  to the goal state ![alt text](formulae/form_35.png "$ y^g $")  <em>during</em> the movement <a class="el" href="citelist.html#CITEREF_kulvicius12joining">[6]</a>. This delayed goal attractor ![alt text](formulae/form_36.png "$ y^{g_d} $")  itself is represented as an exponential dynamical system that starts at ![alt text](formulae/form_34.png "$ y_0 $") , and converges to ![alt text](formulae/form_35.png "$ y^g $")  (in early versions of DMPs, there was no delayed goal system, and ![alt text](formulae/form_36.png "$ y^{g_d} $")  was simply equal to ![alt text](formulae/form_35.png "$ y^g $")  throughout the movement). The combination of these two systems, listed below, leads to a movement that starts and ends with 0 velocities and accelerations, and approximately has a bell-shaped velocity profile. This representation is thus well suited to generating human-like point-to-point movements, which have similar properties.



![alt text](formulae/form_118.png "\begin{eqnarray*} \left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{\mathbf{y}}^{g_d}} \\ {\dot{s}} \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g_d}-\mathbf{y})-\mathbf{z}) + v\cdot f(s))/\tau \\ \mathbf{z}/\tau \\ -\alpha_g({\mathbf{y}^g-\mathbf{y}^{g_d}}) \\ 1/\tau \\ -\alpha_v v (1-v/v_{\mbox{\scriptsize max}}) \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}_0 \\ \mathbf{y}_0 \\ 0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ \mathbf{y}^g \\ 1 \\ 0 \end{array} \right] \end{eqnarray*}") 

![alt text](images/dmp_and_goal_system-svg.png  "A first dynamical movement primitive, with and without a delayed goal system (left: state variable, center: velocities, right: accelerations.")

In my experience, this DMP formulation is the best for learning human-like point-to-point movements (bell-shaped velocity profile, approximately zero velocities and accelerations at beginning and start of the movement), and generates nice normalized data for the function approximator without scaling issues. The image below shows the interactions between the spring-damper system, delayed goal system, phase system and gating system.

![alt text](images/dmpplot_kulvicius2012joining-svg.png  "The various dynamical systems and forcing terms in multi-dimensional DMPs.")

<a name="sec_dmp_summary"></a>

Summary
---------------

The core idea in dynamical movement primitives is to combine dynamical systems, which have nice properties in terms of convergence towards the goal, robustness to perturbations, and independence of time, with function approximators, which allow for the generation of arbitrary (smooth) trajectories. The key enabler to this approach is to gate the output of the function approximator with a gating system, which is 1 at the beginning of the movement, and 0 towards the end.

Further enhancements can be made by making the system autonomous (by using the output of a phase system rather than time as an input to the function approximator), or having initial velocities and accelerations of 0 (by using a delayed goal system).

Multi-dimensional DMPs are achieved by using multi-dimensional dynamical systems, and learning one function approximator for each dimension. Synchronization of the different dimensions is ensure by coupling them with only <em>one</em> phase system. 



<a name="sec_implementation_dmp"></a>

Implementation
---------------

<em> Since a Dynamical Movement Primitive is a dynamical system, the Dmp class derives from the DynamicalSystem class. It overrides the virtual function DynamicalSystem::integrateStart(). Integrating the DMP numerically (Euler or 4th order Runge-Kutta) is done with the generic DynamicalSystem::integrateStep() function. It also implements the pure virtual function DynamicalSystem::analyticalSolution(). Because a DMP cannot be solved analytically (we cannot write it in closed form due to the arbitrary forcing term), calling Dmp::analyticalSolution() in fact performs a numerical Euler integration (although the linear subsystems (phase, gating, etc.) are analytically solved because this is faster computationally).</em>

<em>Please note that in this tutorial we have used the notation ![alt text](formulae/form_11.png "$[z~y]$")  for consistency with the DMP literature. In the C++ implementation, the order is rather ![alt text](formulae/form_24.png "$[y~z]$") .</em>

<em><em>Remark</em>. Dmp inherits the function DynamicalSystem::integrateStep() from the DynamicalSystem class. DynamicalSystem::integrateStep() uses either Euler integration, or 4-th order Runge-Kutta. The latter is more accurate, but requires 4 calls of DynamicalSystem::differentialEquation() instead of 1). Which one is used can be set with DynamicalSystem::set_integration_method(). To numerically integrate a dynamical system, one must carefully choose the integration time dt. Choosing it too low leads to inaccurate integration, and the numerical integration will diverge from the 'true' solution acquired through analytical solution. See <a href="http://en.wikipedia.org/wiki/Euler%27s_method">http://en.wikipedia.org/wiki/Euler%27s_method</a> for examples. Choosing dt depends entirely on the time-scale (seconds vs. years) and parameters of the dynamical system (time constant, decay parameters). For DMPs, which are expected to take between 0.5-10 seconds, dt is usually chosen to be in the range 0.01-0.001. </em>


