import numpy as np
import fxc as gdb



# ### Definition of the Fermi wave number for the unpolarized electron gas
def qf(rs):
	return np.power(9.0*0.25*np.pi, 1.0/3.0) / rs


# ### Analytical Parametrization of the same-spin component of the ontop pair-distribution function g(0) taken from Dornheim et al [PRL 125 (23), 235001 (2020)]
def OnTop(rs,t):
	
	# Fit coefficients taken from ground-state QMC study by Spink et al [Phys Rev B 88, 085121 (2013)]
	a_Spink = 0.18315
	b_Spink = -0.0784043
	c_Spink = 1.02232
	d_Spink = 0.0837741
	
	# Finite-T fit coefficients obtained from fit to restricted PIMC data by Brown et al [PRL 110, 146405 (2013)]
	alpha_1_a      = 18.4377      
	beta_1_a       = 24.1339      
	beta_2_a       = 1.86499      
	alpha_1_b      = -0.24368     
	beta_1_b       = 0.252577     
	beta_2_b       = 0.127043     
	alpha_1_c      = 2.23663      
	beta_1_c       = 0.445526     
	beta_2_c       = 0.408504     
	alpha_2_c      = 0.448937     
	alpha_1_d      = 0.0589015    
	beta_1_d       = -0.598508    
	beta_2_d       = 0.513162     
	
	return ( 1.0 + ( a_Spink + alpha_1_a * t  ) / ( 1.0 + t* beta_1_a + t*t*t *beta_2_a ) * np.sqrt(rs) + ( b_Spink + alpha_1_b * np.sqrt(t)  ) / ( 1.0 + t* beta_1_b + t*t *beta_2_b ) * rs ) / ( 1.0 + ( c_Spink + alpha_1_c * np.sqrt(t) + alpha_2_c*t*np.sqrt(t)  ) / ( 1.0 + t* beta_1_c + t*t *beta_2_c ) * rs + ( d_Spink + alpha_1_d * np.sqrt(t)  ) / ( 1.0 + t* beta_1_d + t*t *beta_2_d ) * rs*rs*rs )



# ### Definition of the activation function used in the ESA
# x: q/q_F (q_F is Fermi wave number, see above)
# a: x_m
# b: eta
def Activation(x,a,b):
	return 0.5*( np.tanh( b*(x-a) ) + 1.0 )



# ### Implementation of the theta-dependent ESA parameter x_m(theta), for the activation function
def x_m(theta):
	A_x = 2.64
	B_x = 0.31
	C_x = 0.08
	return A_x + B_x*theta + C_x*theta**2











# ### Local Field Correction within the Effective Static Approximation (ESA):
# x=q/qF
def G_ESA( x, xm, eta, rs, theta ):
	
	# obtain the exact static limit of the LFC from the neural net:
	G_ML = G(x,rs,theta)
	if x > 5.0:
		G_ML = G(5.0,rs,theta)
	
	# obtain the value of the total 
	onTop = 0.5*OnTop( rs, theta ) 
	
	# compute the activation function at these parameters
	A = Activation(x,xm,eta)
	
	# compute the consistent limit for infinite wave-number
	Ginfty = 1.0 - onTop
	
	# final result for G_ESA:
	result = A*Ginfty + (1.0-A)*G_ML
	
	return result











# ### Fitted coefficients for the analytical parametrization of G(q;rs,t) within the ESA

ABCD = [ 0.66477593, -4.59280227,  1.24649624, -1.27089927,  1.26706839, -0.4327608,
  2.09717766,  1.15424724, -0.65356955, -1.0206202,   5.16041218, -0.23880981,
  1.07356921, -1.67311761,  0.58928105,  0.8469662,   1.54029035, -0.71145445,
 -2.31252076,  5.83181391,  2.29489749,  1.76614589, -0.09710839, -0.33180686,
  0.56560236,  1.10948188, -0.43213648,  1.3742155,  -4.01393906, -1.65187145,
 -1.75381153, -1.17022854,  0.76772906,  0.63867766,  1.07863273, -0.35630091]


# ### Functional form for q-dependence of G_ESA
# y=[x,rs,theta]
# x=q/q_F
def G_fit_wrap_extended(y, alpha, beta, gamma, delta):
	my_x = y[:,0]
	myRS = y[:,1]
	myTHETA = y[:,2]
	
	# compute the first part of ESA, i.e., fit to the neural-net representation [Dornheim et al, J. Chem. Phys. 151, 194104 (2019)] of the static LFC
	# gdb.Groth_A gives pre-factor to the exact compressibility sum-rule (CSR) computed from the prametrization of fxc by Groth et al [PRL 119 (13), 135001 (2017)]
	G_ML_fit = gdb.Groth_A(myRS,myTHETA)*my_x*my_x * (1.0 + alpha*my_x + beta*my_x**0.5) / (1.0+ gamma*my_x + delta * my_x**1.25 + gdb.Groth_A(myRS,myTHETA)*my_x**2)
	
	# Obtain the value of the full ontop PDF g(0). Factor 0.5, because, OnTop returns only same-spin component
	onTop = 0.5*OnTop( myRS, myTHETA )
	
	# consistent large-q limit of an effectively static theory for the LFC
	Ginfty = 1.0 - onTop
	
	# width of transition between limits in the activation function is constant in this analytical representation
	ETA = 3.0
	
	# the position (wave-number) of the transition depends on theta only and has been parametrized above
	XM = x_m(myTHETA)
	
	# the final result for the LFC within ESA is given by the combination of the static LFC (G_ML) and the large-g limit (Ginfty), connected by the Activation function
	A = Activation(my_x,XM,ETA)
	return  G_ML_fit*(1.0-A) + A*Ginfty



# ### rs-dependent representation of the four parameters in G_fit_wrap_extended(y, alpha, beta, gamma, delta)
def alpha_extended(rs, a, b, c):
	return (a+b*rs)/(1.0+c*rs)

def beta_extended(rs, a, b, c):
	return (a+b*rs)/(1.0+c*rs)

def gamma_extended(rs, a, b, c):
	return (a+b*rs)/(1.0+c*rs)

def delta_extended(rs, a, b, c):
	return (a+b*rs)/(1.0+c*rs)



# ### theta-dependence of the parameters in alpha_extended, beta_extended, ...
def f_extended(t, a, b, c):
	return a + b*t + c*t**1.5


# ### Analytical representation of the static LFC within ESA
# y = [x,rs,theta]
# x = q/q_F
# coeff are fitted coefficients, see ABCD above
def G_analytical(y, coeff):
	my_x = y[:,0]
	myRS = y[:,1]
	myTHETA = y[:,2]
	
	# ### Determination of first rs-parameter alpha
	
	a = coeff[0]
	b = coeff[1]
	c = coeff[2]
	my_alpha_a = f_extended(myTHETA, a, b, c)
	
	a = coeff[3]
	b = coeff[4]
	c = coeff[5]
	my_alpha_b = f_extended(myTHETA, a, b, c)
	
	a = coeff[6]
	b = coeff[7]
	c = coeff[8]
	my_alpha_c = f_extended(myTHETA, a, b, c)
	
	my_alpha = alpha_extended( myRS, my_alpha_a, my_alpha_b, my_alpha_c )
	
	
	
	
	# ### Determination of second rs-parameter beta
	
	a = coeff[9]
	b = coeff[10]
	c = coeff[11]
	my_beta_a = f_extended(myTHETA, a, b, c)
	
	a = coeff[12]
	b = coeff[13]
	c = coeff[14]
	my_beta_b = f_extended(myTHETA, a, b, c)
	
	a = coeff[15]
	b = coeff[16]
	c = coeff[17]
	my_beta_c = f_extended(myTHETA, a, b, c)
	
	my_beta = beta_extended( myRS, my_beta_a, my_beta_b, my_beta_c )
	
	
	
	# ### Determination of third rs-parameter gamma
	
	a = coeff[18]
	b = coeff[19]
	c = coeff[20]
	my_gamma_a = f_extended(myTHETA, a, b, c)
	
	a = coeff[21]
	b = coeff[22]
	c = coeff[23]
	my_gamma_b = f_extended(myTHETA, a, b, c)
	
	a = coeff[24]
	b = coeff[25]
	c = coeff[26]
	my_gamma_c = f_extended(myTHETA, a, b, c)
	
	my_gamma = gamma_extended( myRS, my_gamma_a, my_gamma_b, my_gamma_c )
	
	
	
	# ### Determination of fourth rs-parameter delta
	
	a = coeff[27]
	b = coeff[28]
	c = coeff[29]
	my_delta_a = f_extended(myTHETA, a, b, c)
	
	a = coeff[30]
	b = coeff[31]
	c = coeff[32]
	my_delta_b = f_extended(myTHETA, a, b, c)
	
	a = coeff[33]
	b = coeff[34]
	c = coeff[35]
	my_delta_c = f_extended(myTHETA, a, b, c)
	
	my_delta = delta_extended( myRS, my_delta_a, my_delta_b, my_delta_c )


	return_value = G_fit_wrap_extended(y, my_alpha, my_beta, my_gamma, my_delta)
	return return_value








# ### Example value:

print( "Example value: G_ESA(2.6,10.0,2.0) = ", G_analytical(np.array([[2.6,10.0,2.0]]),ABCD)[0] )


















