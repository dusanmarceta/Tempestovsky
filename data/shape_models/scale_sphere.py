from stl import mesh

# učitaj STL
sphere = mesh.Mesh.from_file('500m_ico_sphere_1280_facets.stl')

# skaliranje
sphere.vectors *= 0.1

# snimi novi STL
sphere.save('50m_ico_sphere_1280_facets.stl')