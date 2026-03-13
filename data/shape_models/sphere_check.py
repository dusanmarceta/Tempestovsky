import numpy as np
from stl import mesh

sphere = mesh.Mesh.from_file('500m_ico_sphere_80_facets.stl')

# sva temena
vertices = sphere.vectors.reshape(-1, 3)

# centar sfere (ako nije tačno u (0,0,0))
center = vertices.mean(axis=0)

# rastojanja od centra
r = np.linalg.norm(vertices - center, axis=1)

# radijus (srednja vrednost)
radius = r.mean()

print(radius)





volume, cog, inertia = sphere.get_mass_properties()

radijus1 = (3*volume / 4 / np.pi)**(1/3)

print(volume)
print(radijus1)