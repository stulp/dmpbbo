Function Approximation
===============

This module implements a set of function approximators, i.e. supervised learning algorithms that are **trained** with demonstration pairs input/target, after which they **predict** output values for new inputs. For simplicity, DmpBbo focusses on batch learning (not incremental), as the main use cases in the context of dmpbbo is imitation learning.

<a name="sec_fa_metaparameters"></a>

Meta-parameters and model-parameters
---------------

In this module, algorithmic parameters are called meta-parameters, and the parameters of the model when the function approximator has been trained are called model-parameters. The rationale for this is that an untrained function approximator can be entirely reconstructed if its meta-parameters are known; this is useful for saving to file and making copies. A trained function approximator can be compeletely reconstructed given only its model-parameters.

The life-cycle of a function approximator is as follows:

1. **Initialization.** The function approximator is initialized with meta-parameters.

2. **Training.** The function approximator is trained, which performs the mapping: ![alt text](formulae/form_84.png "$ \mbox{train}: \mbox{MetaParameters} \times \mbox{Inputs} \times \mbox{Targets} \mapsto \mbox{ModelParameters} $") 

3. **Prediction.** The function approximator predicts, which performs the mapping: ![alt text](formulae/form_85.png "$ \mbox{predict}: \mbox{ModelParameters} \times \mbox{Input} \mapsto \mbox{Output}$") 


<a name="sec_fa_unified_model"></a>

Unified Model for Function Approximators
---------------

The unified model is a unified representation for the different model parameters used by the different function approximators.

Whilst coding this library and numerous discussion with Olivier Sigaud, it became apparent that the latent function representations of all the function approximators in this library all use the same generic model. Each specific model (i.e. as used in GPR, GMR, LWR, etc.) is a special case of the Unified Model. We discuss this in the paper titled: "Many Regression Algorithms, One Unified Model - A Review, Freek Stulp and Olivier Sigaud", which you should be able to find in an on-line search. 

Further reading
---------------

The next tutorials to go to would be:
* <a href="dmp.md">Dynamical Movement Primitives</a> (if you already know about dynamical systems)
* <a href="dynamicalsystems.md">Dynamical Systems</a> (if you don't)

