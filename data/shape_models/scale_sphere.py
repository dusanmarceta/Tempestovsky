from stl import mesh

# učitaj STL
sphere = mesh.Mesh.from_file('500m_ico_sphere_80_facets.stl')

# skaliranje
sphere.vectors *= 1.56

# snimi novi STL
sphere.save('didymos_80_facets.stl')