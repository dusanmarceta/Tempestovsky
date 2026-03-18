import numpy as np
from stl import mesh

# učitaj STL
mesh_file = mesh.Mesh.from_file('didymos_80_facets.stl')

# temena svih trouglova
v0 = mesh_file.vectors[:,0]
v1 = mesh_file.vectors[:,1]
v2 = mesh_file.vectors[:,2]

# centri faceta
positions = (v0 + v1 + v2) / 3.0  # shape (N,3)

# normalne
normals = mesh_file.normals.copy()  # shape (N,3)

# površina svakog trougla
areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

# centar mesh-a
center = positions.mean(axis=0)
positions_centered = positions - center

# normalizacija normala
normals_normalized = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

# okreni normale koje gledaju unutra
dot_products = np.einsum('ij,ij->i', normals_normalized, positions_centered)
normals_corrected = normals_normalized.copy()
normals_corrected[dot_products < 0] *= -1

# formula za zapreminu
volume = (1.0 / 3.0) * np.sum(
    areas * np.einsum('ij,ij->i', positions_centered, normals_corrected)
)

print("Zapremina iz formule:", abs(volume))


volume_original, cog, inertia = mesh_file.get_mass_properties()

print("Njihova zapremina", volume_original)

r = (3*volume / 4 / np.pi)**(1/3)

