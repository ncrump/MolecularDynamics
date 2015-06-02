"""
Created on Sun Feb 08 09:11:52 2015
"""

# Plot of Lennard-Jones potential

import numpy as np
import matplotlib.pyplot as plt

sig = 1.0
r0 = 1.12246*sig

R = []
P = []
F = []
for r in np.linspace(0.98*sig,1.8*r0,1000):
    R.append(r/sig)
    P.append(4*(((sig/r)**12) - ((sig/r)**6)))
    F.append((4*r)*((12*(sig/r)**12) - (6*(sig/r)**6)))

plt.figure()
plt.subplot(2,1,1)
plt.plot(R,P,label='LJ Potential')
plt.ylabel(r'U/$\epsilon$')
plt.legend()
plt.subplot(2,1,2)
plt.plot(R,F,label='LJ Force')
plt.ylabel(r'F/$\epsilon$')
plt.xlabel(r'R/$\sigma$')
plt.legend()