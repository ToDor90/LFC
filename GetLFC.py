import numpy as np
import matplotlib.pyplot as plt

from keras import losses
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

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





# ### Example value:

print( "Example value: G(1.2,1.9,3.3) = ", G(1.2,1.9,3.3) )





# ### Plot the q-dependence of the static LFC for a few examples (x=q/q_F)
RS1 = 2.0
THETA1 = 1.0

RS2 = 0.7
THETA2 = 3.5

RS3 = 10.0
THETA3 = 0.5




label1 = "rs=" + str(RS1) + ", t=" + str(THETA1)
label2 = "rs=" + str(RS2) + ", t=" + str(THETA2)
label3 = "rs=" + str(RS3) + ", t=" + str(THETA3)

x_values = [(0.5+i)*5.0/600.0 for i in range(600)]

G_values = [G(x,RS1,THETA1) for x in x_values]
G_values2 = [G(x,RS2,THETA2) for x in x_values]
G_values3 = [G(x,RS3,THETA3) for x in x_values]

print("#############################")




plt.figure(figsize=(8,8))

plt.plot(x_values, G_values, linewidth=2, color='r', label=label1)
plt.plot(x_values, G_values2, linewidth=2, color='b', label=label2)
plt.plot(x_values, G_values3, linewidth=2, color='g', label=label3)
plt.title('The static local field correction')
plt.ylabel('G')
plt.xlabel('x=q/q_F')

#plt.ylim(bottom=0,top=3)
plt.xlim(0,5)
plt.legend()

plt.show()


