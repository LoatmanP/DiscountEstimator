# DiscountEstimator
A python program for estimating discount rates from several delay discounting models

Built to be compatible with the scikit-learn library

Below are each model and the corresponding choice function used to estimate parameters.

Exponential Discounting: Vd=V0 e^(-kd), L(α(x2e^kt2 - x1e^kt1)), Samuelson (1937)

Hyperbolic Discounting: Vd=  V0/(1+kD), L(α(x2/(1+Kt2) - x1/(1+kt1))), Mazur (1987)

Generalized Hyperbolic Discounting: Vd= V0[1+θd]^(-k/θ), L(α(x2[1+θt2]^(-k/θ) - x1[1+θt1]^(-k/θ) )), Lowenstein & Prelec (1992)

Hyperboloid Discounting: Vd=V0/(1+kD)^s, L(α( (x2/(1+Kt2)^s - (x1/(1+Kt1)^s )), Green & Myerson (2004)

Hyperbolic with Exponent Discounting: Vd=V0/1+KD^s, L(α (x2/1+Kt2^s -(x1/1+Kt1^s )), Rogriguez & Logue (1988)

Quasi-Hyperbolic Discounting: Vd= V0δe^(-kd), L(α(x2δe^(-kt2) - x1δe^(-kt1))), Laibson (1997)

-----------------------------------------------------------------------------------------------------------

# Methods

*fit*(self, X, y):
  X = smaller, sooner reward/delay; larger, later reward/delay
  y = choices (0,1)
  optimization routine is scipy brute force with fmin finish
  return self
 
*sse*(self, params, X,y):
X = smaller, sooner reward/delay; larger, later reward/delay
y = choices (0,1)
returns sse (sum of squared error)

*choice* (self, X, params=[]):
X = smaller, sooner reward/delay; larger, later reward/delay
returns pLL (probability of selecting the larger, later choice item)

*predict_proba*(self, X, params =[]):
returns the predicted probability of selecting the lager, later choice item

*predict*(self, X, params=[]):
returns the predicted outcome (0,1)

*set_params*(self):
returns the initialized parameters (e.g., discount rate, rho)

*get_params*(self):
returns the parameters (e.g., discount rate, rho)

