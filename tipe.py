import numpy as np
N = 200
#(y, x)

tableau = np.zeros((N, N))
tableau[0,2] = 0

# Centre des bobines
c_bobines = [(int(N/4), int(N/4)), (int(3*N/4), int(N/4)), (int(N/4), int(3*N/4)), (int(3*N/4), int(3*N/4))]
rayons_bobines = [N/6] * len(c_bobines)

for centre in c_bobines:
    c_y, c_x = centre
    tableau[c_y, c_x] = 1

def intersection_cercles(c1, r, c2, r_prime):
    intersection = 0
    for i in range(N):
        for j in range(N):
            if (i - c1[0])**2 + (j - c1[1])**2 <= r**2 and (i - c2[0])**2 + (j - c2[1])**2 <= r_prime**2:
                intersection += 1

    return intersection

class Aimant():
    def __init__(self, r, theta, rayon):
        self.r = r
        self.theta = theta
        self.rayon = rayon

    def cartesien(self):
        return (N/2 + self.r * np.cos(self.theta), N/2 + self.r * np.sin(self.theta))
    
aimant = Aimant(N/3, np.pi, N/20)

nb_pixels_aimant = 0
for i in range(N):
    for j in range(N):
        if (i - aimant.cartesien()[0])**2 + (j - aimant.cartesien()[1])**2 <= aimant.rayon**2:
            nb_pixels_aimant += 1

resultats = [[]] * len(c_bobines)
theta_space = np.linspace(0, 2 * np.pi, 100)
for theta in theta_space:
    aimant.theta = theta
    somme = [intersection_cercles(bobine, rayon, aimant.cartesien(), aimant.rayon) for bobine, rayon in zip(c_bobines, rayons_bobines)]
    for bobine, rayon in zip(c_bobines, rayons_bobines):
        resultats[c_bobines.index(bobine)].append(intersection_cercles(bobine, rayon, aimant.cartesien(), aimant.rayon))
# Draw matplotlib all the circles
    
aimant_points = [(i, j) for i in range(N) for j in range(N) if (i - aimant.cartesien()[0])**2 + (j - aimant.cartesien()[1])**2 <= aimant.rayon**2]
bobines_points = []
for bobine, rayon in zip(c_bobines, rayons_bobines):
    for i in range(N):
        for j in range(N):
            if (i - bobine[0])**2 + (j - bobine[1])**2 <= rayon**2:
                bobines_points.append((i, j))

# Parametres physiques
surface_aimant = 25 # cm^2
surface_bobine = 100 # cm^2

def calculer_flux():
    pass

def calculer_fem():
    pass

def calculer_ueff():
    pass

import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

# Plot the first graph
for r in resultats:
    ax1.plot(theta_space, r)
ax1.set_title('Proportion couverture bobine aimant')

# Second graph
ax2.imshow(tableau, cmap='gray')
ax2.scatter([i[1] for i in bobines_points], [i[0] for i in bobines_points], c='r', s=100, label='Bobines')
ax2.scatter([i[1] for i in aimant_points], [i[0] for i in aimant_points], c='b', s=100, label='Aimant')

ax2.set_title('Schematisation')
ax2.legend()

plt.show()
