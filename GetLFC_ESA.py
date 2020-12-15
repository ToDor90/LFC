import numpy as np
import matplotlib.pyplot as plt

from keras import losses
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

import mpmath

LR = LeakyReLU()
LR.__name__ = 'relu'




# ### Define the Keras model

N_LAYER = 40
W_LAYER = 64

model = Sequential()
model.add(Dense(W_LAYER, input_dim=3, activation=LR))

REGULARIZATION_RATE = 0.0000006

for i in range(N_LAYER-1):
	model.add( Dense( W_LAYER, activation=LR, kernel_regularizer=regularizers.l2( REGULARIZATION_RATE ) ) )

model.add(Dense(1, activation='linear'))




# ### Load the trained weights from hdf5 file

model.load_weights('LFC.h5')




# ### Define simple wrapper function (x=q/q_F):


def G(x,rs,theta):
	result = model.predict( np.array( [[x,rs,theta]] ) )
	return result[0][0]


def qf(rs):
	return np.power(9.0*0.25*np.pi, 1.0/3.0) / rs



###############################################
# ### New definitions for the ESA
###############################################

# ### Activation function
def Activation(x,xm,eta):
	return 0.5*( mpmath.tanh( eta*(x-xm) ) + 1.0 )


# ### Spin-up-down component of the pair distribution function at zero distance, g(0)
def OnTop(rs,theta):
	t = theta
	
	# parameters from the ground-state parametrization by Spink et al. [Phys. Rev. B 88, 085121 (2013)]
	a_Spink = 0.18315
	b_Spink = -0.0784043
	c_Spink = 1.02232
	d_Spink = 0.0837741
	
	# parameters for the temperature-dependent parametrization:
	Qalpha_1_a      = 18.4377      
	Qbeta_1_a       = 24.1339      
	Qbeta_2_a       = 1.86499      
	Qalpha_1_b      = -0.24368     
	Qbeta_1_b       = 0.252577     
	Qbeta_2_b       = 0.127043     
	Qalpha_1_c      = 2.23663      
	Qbeta_1_c       = 0.445526     
	Qbeta_2_c       = 0.408504     
	Qalpha_2_c      = 0.448937     
	Qalpha_1_d      = 0.0589015    
	Qbeta_1_d       = -0.598508    
	Qbeta_2_d       = 0.513162     

	return ( 1.0 + ( a_Spink + Qalpha_1_a * t  ) / ( 1.0 + t* Qbeta_1_a + t*t*t *Qbeta_2_a ) * np.sqrt(rs) + ( b_Spink + Qalpha_1_b * np.sqrt(t)  ) / ( 1.0 + t* Qbeta_1_b + t*t *Qbeta_2_b ) * rs ) / ( 1.0 + ( c_Spink + Qalpha_1_c * np.sqrt(t) + Qalpha_2_c*t*np.sqrt(t)  ) / ( 1.0 + t* Qbeta_1_c + t*t *Qbeta_2_c ) * rs + ( d_Spink + Qalpha_1_d * np.sqrt(t)  ) / ( 1.0 + t* Qbeta_1_d + t*t *Qbeta_2_d ) * rs*rs*rs )



# ### Local Field Correction within the Effective Static Approximation (ESA):
# x=q/qF
def G_ESA( x, xm, eta, rs, theta ):
	
	# obtain the exact static limit of the LFC from the neural net:
	G_ML = G(x,rs,theta)
	
	# obtain the value of the total ontop pdf
	onTop = 0.5*OnTop( rs, theta ) 
	
	# compute the activation function at these parameters
	A = Activation(x,xm,eta)
	
	# compute the consistent limit for infinite wave-number
	Ginfty = 1.0 - onTop
	
	# final result for G_ESA:
	result = A*Ginfty + (1.0-A)*G_ML
	
	return result




# ### Plot the q-dependence of the static LFC for a few examples (x=q/q_F)

# Example density and temperature
RS = 6.0
THETA = 0.5

# Reasonable parameters for the activation function:
XM = 3.0
ETA = 3.0


label = "G: neural net"
label_ESA = "G: ESA"

x_values = [(0.5+i)*5.0/600.0 for i in range(600)]

G_values = [G(x,RS,THETA) for x in x_values]
G_values_ESA = [G_ESA(x, XM, ETA, RS, THETA) for x in x_values]


print("#############################")




plt.figure(figsize=(8,8))

plt.plot(x_values, G_values, linewidth=2, color='b', label=label)
plt.plot(x_values, G_values_ESA, linewidth=2, color='r', label=label_ESA)
plt.title('The static local field correction at rs=' + str(RS) + ' and theta=' + str(THETA))
plt.ylabel('G')
plt.xlabel('x=q/q_F')

#plt.ylim(bottom=0,top=3)
plt.xlim(0,5)
plt.legend()

plt.show()


