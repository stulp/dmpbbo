Dynamical Systems
===============

Let a *state* be a vector of real numbers. A dynamical system consists of such a state and a rule that describes how this state will change over time; it describes what future state follows from the current state. A typical example is radioactive decay, where the state ![alt text](formulae/form_39.png "$x$")  is the number of atoms, and the rate of decay is ![alt text](formulae/form_40.png "$\frac{dx}{dt}$")  proportional to ![alt text](formulae/form_39.png "$x$") : ![alt text](formulae/form_41.png "$ \frac{dx}{dt} = -\alpha x$") . Here, ![alt text](formulae/form_7.png "$\alpha$")  is the *decay constant* and ![alt text](formulae/form_42.png "$\dot{x}$")  is a shorthand for ![alt text](formulae/form_40.png "$\frac{dx}{dt}$") . Such an evolution rule describes an implicit relation between the current state ![alt text](formulae/form_43.png "$ x(t) $")  and the state a short time in the future ![alt text](formulae/form_44.png "$x(t+dt)$") .

If we know the initial state of a dynamical system, e.g. ![alt text](formulae/form_45.png "$x_0\equiv x(0)=4$") , we may compute the evolution of the state over time through *numerical integration*. This means we take the initial state ![alt text](formulae/form_46.png "$ x_0$") , and iteratively compute subsequent states ![alt text](formulae/form_44.png "$x(t+dt)$")  by computing the rate of change ![alt text](formulae/form_42.png "$\dot{x}$") , and integrating this over the small time interval ![alt text](formulae/form_47.png "$dt$") . A pseudo-code example is shown below for ![alt text](formulae/form_45.png "$x_0\equiv x(0)=4$") , ![alt text](formulae/form_48.png "$dt=0.01s$")  and ![alt text](formulae/form_49.png "$\alpha=6$") .

	alpha=6; // Decay constant
	dt=0.01; // Duration of one integration step
	x=4.0;   // Initial state
	t=0.0;   // Initial time
	while (t<1.5) {
		dx = -alpha*x; // Dynamical system rule
		x = x + dx*dt; // Project x into the future for
		               // a small time step dt (Euler integration)
		t = t + dt;    // The future is now!
	}
                           
This procedure is called *integrating the system*, and leads the trajectory plotted below (shown for both ![alt text](formulae/form_49.png "$\alpha=6$")  and ![alt text](formulae/form_50.png "$\alpha=3$") .


![alt text](images/exponential_decay-svg.png  "Evolution of the exponential dynamical system.")

The evolution of many dynamical systems can also be determined analytically, by explicitly solving the differential equation. For instance, ![alt text](formulae/form_51.png "$N(t) = x_0e^{-\alpha t}$")  is the solution to ![alt text](formulae/form_52.png "$\dot{x} = -\alpha x$") . Why? Let's plug ![alt text](formulae/form_53.png "$x(t) = x_0e^{-\alpha t}$")  into ![alt text](formulae/form_54.png "$\frac{dx}{dt} = -\alpha x$") , which leads to ![alt text](formulae/form_55.png "$\frac{d}{dt}(x_0e^{-\alpha t}) = -\alpha (x_0e^{-\alpha t})$") . Then derive the left side of the equations, which yields ![alt text](formulae/form_56.png "$-\alpha(x_0e^{-\alpha t}) = -\alpha (x_0e^{-\alpha t})$") . QED. Note that the solution works for arbitrary ![alt text](formulae/form_57.png "$x_0$") . It should, because the solution should not depend on the initial state.




<a name="sec_dyn_sys_properties"></a>

Properties and Features of Linear Dynamical Systems
---------------


<a name="sec_dyn_sys_convergence"></a>
### Convergence towards the Attractor

In the limit of time, the dynamical system for exponential decay will converge to 0 (i.e. ![alt text](formulae/form_58.png "$x(\infty) = x_0e^{-\alpha\infty} = 0$") ). The value 0 is known as the *attractor* of the system. For simple dynamical systems, it is possible to *prove* that they will converge towards the attractor.

Suppose that the attractor state in our running example is not 0, but 1. In that case, we change the attractor state of the exponential decay to ![alt text](formulae/form_59.png "$x^g$")  ( ![alt text](formulae/form_60.png "$g$") =goal) and define the following differential equation: 


![alt text](formulae/form_61.png "\begin{eqnarray*} \dot{x} =&amp; -\alpha(x-x^g) &amp; \mbox{~with attractor } x^g \end{eqnarray*}") 


This system will now converge to the attractor state ![alt text](formulae/form_59.png "$x^g$") , rather than 0.


![alt text](images/change_tau_attr-svg.png "Changing the attractor state or time constant.") 


<a name="sec_dyn_sys_perturbations"></a>
### Robustness to Perturbations

Another nice feature of dynamical systems is their robustness to perturbations, which means that they will converge towards the attractor even if they are perturbed. The figure below shows how the perturbed system (cyan) converges towards the attractor state just as the unperturbed system (blue) does.

![alt text](images/perturb-svg.png "Perturbing the dynamical system.") 


<a name="sec_dyn_sys_time_constant"></a>
### Changing the speed of convergence: The time constant

The rates of change computed by the differential equation can be increased or decreased (leading to a faster or slower convergence) with a *time constant*, which is usually written as follows:



![alt text](formulae/form_62.png "\begin{eqnarray*} \tau\dot{x} =&amp; -\alpha(x-x^g)\\ \dot{x} =&amp; (-\alpha(x-x^g))/\tau \end{eqnarray*}") 


*Remark*. For an exponential system, decreasing the time constant ![alt text](formulae/form_63.png "$\tau$")  has the same effect as increasing ![alt text](formulae/form_7.png "$\alpha$") . For more complex dynamical systems with several parameters, it is useful to have a separate parameter that changes only the speed of convergence, whilst leaving the other parameters the same.


<a name="sec_dyn_sys_multi"></a>
### Multi-dimensional states

The state ![alt text](formulae/form_39.png "$x$")  need not be a scalar, but may be a vector. This then represents a multi-dimensional state, i.e. ![alt text](formulae/form_64.png "$\tau\dot{\mathbf{x}} = -\alpha(\mathbf{x}-\mathbf{x}^g)$") . In the code, the size of the state vector ![alt text](formulae/form_65.png "$dim(\mathbf{x})\equiv dim(\dot{\mathbf{x}})$")  of a dynamical system is returned by the function DynamicalSystem::dim()


<a name="sec_dyn_sys_autonomy"></a>
### Autonomy

Dynamical system that do not depend on time are called *autonomous*. For instance, the formula ![alt text](formulae/form_66.png "$ \dot{x} = -\alpha x$")  does not depend on time, which means the exponential system is autonomous.


<a name="sec_further_dyn_sys"></a>
### Further examples of dynamical systems

Apart from the [exponential system]( http://en.wikipedia.org/wiki/Exponential_decay) described above,  further (first order) linear dynamical systems that are implemented in dmpbbo include a [sigmoid system](http://en.wikipedia.org/wiki/Sigmoid_function), as well as a dynamical system that has a constant velocity (TimeSystem), so as to mimic the passing of time (time moves at a constant rate per time ;-)


![alt text](formulae/form_67.png "\begin{eqnarray*} \dot{x} =&amp; -\alpha (x-x^g) &amp; \mbox{exponential decay/growth} \label{equ_}\\ \dot{x} =&amp; \alpha x (\beta-x) &amp; \mbox{sigmoid} \label{equ_}\\ \dot{x} =&amp; 1/\tau &amp; \mbox{constant velocity (mimics the passage of time)} \label{equ_}\\ \end{eqnarray*}") 


![alt text](images/sigmoid-svg.png "Exponential (blue) and sigmoid (purple) dynamical systems.") 

<a name="dyn_sys_second_order_systems"></a>
## Second-Order Systems

The <b>order</b> of a dynamical system is the order of the highest derivative in the differential equation. For instance, ![alt text](formulae/form_52.png "$\dot{x} = -\alpha x$")  is of order 1, because the derivative with the highest order ( ![alt text](formulae/form_42.png "$\dot{x}$") ) has order 1. Such a system is known as a first-order system. All systems considered so far have been first-order systems, because the derivative with the highest order, i.e. ![alt text](formulae/form_68.png "$ \dot{x} $") , has always been of order 1.


<a name="dyn_sys_spring_damper"></a>
### Spring-Damper Systems

An example of a second order system (which also has terms ![alt text](formulae/form_69.png "$ \ddot{x} $") ) is a [spring-damper system]( http://en.wikipedia.org/wiki/Damped_spring-mass_system), where ![alt text](formulae/form_70.png "$k$")  is the spring constant, ![alt text](formulae/form_71.png "$c$")  is the damping coefficient, and ![alt text](formulae/form_72.png "$m$")  is the mass:




![alt text](formulae/form_73.png "\begin{eqnarray*} m\ddot{x}=&amp; -kx -c\dot{x} &amp; \mbox{spring-damper (2nd order system)} \label{equ_}\\ \ddot{x}=&amp; (-kx -c\dot{x})/m &amp; \end{eqnarray*}") 



<a name="dyn_sys_critical_damping"></a>
#### Critical Damping

A spring-damper system is called critically damped when it converges to the attractor as quickly as possible without overshooting, as the red plot in 
[this graph](http://en.wikipedia.org/wiki/File:Damping_1.svg).

Critical damping occurs when ![alt text](formulae/form_3.png "$c = 2\sqrt{mk}$") .


<a name="dyn_sys_rewrite_second_first"></a>
### Rewriting one 2nd Order Systems as two 1st Order Systems

For implementation purposes, it is more convenient to work only with 1st order systems. Fortunately, we can expand the state ![alt text](formulae/form_74.png "$ x $")  into two components ![alt text](formulae/form_75.png "$ x = [y~z]^T$")  with ![alt text](formulae/form_76.png "$ z = \dot{y}$") , and rewrite the differential equation as follows:

![alt text](formulae/form_77.png "$ \left[ \begin{array}{l} \dot{y} \\ \dot{z} \end{array} \right] = \left[ \begin{array}{l} z \\ (-ky -cz)/m \end{array} \right] $") 

With this rewrite, the left term contains only first order derivatives, and the right term does not contain any derivatives. This is thus a first order system. Integrating such an expanded system is done just as one would integrate a dynamical system with a multi-dimensional state.


## Further reading

The next tutorials to go to would be:
* <a href="dmp.md">Dynamical Movement Primitives</a> (if you already know about function approximation)
* <a href="functionapproximators.md">Function Approximation</a> (if you don't)


