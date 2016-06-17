import numpy as np
import evol
from stl import mesh

class Model():
    def __init__(self, model = 'z40_sn', directory = '/media/conrad/DATA1/output/', index = 400):
        self.index = index

        self.e = evol.Evol(model, directory)
        self.e.initialize_step(index)

    def height_map(self, type = 'rshock'):
        if type == 'rshock':
            rshock = self.e.shock_radius(self.index)
            return rshock

    def create_stl(self, filename = '/home/conrad/source/sn_analysis/3dprint/output.stl'):
        phi     = self.e.data.yzn()
        theta   = self.e.data.zzn()
        r       = self.height_map()

        # normalise r
        r = r / np.mean(r)

        nphi    = len(phi)
        ntheta  = len(theta)

        vertices = np.ndarray((nphi, ntheta, 3)) # 3 coordinates per vertex
        vertex_id = np.ndarray((nphi, ntheta)) # identifier of the vertex
        vid = 0
        for i in range(nphi):
            for j in range(ntheta):
                vertices[i,j,0] = r[i,j]
                vertices[i,j,1] = phi[i]
                vertices[i,j,2] = theta[j]
                vertex_id[i,j]  = vid
                vid = vid + 1

        poles = np.ndarray((2, 3)) # 0:north 1:south
        poles[0,0] = np.mean(r[0,:])
        poles[0,1] = 0
        poles[0,2] = 0
        poles[1,0] = np.mean(r[-1,:])
        poles[1,1] = np.pi
        poles[1,2] = 0

        faces = np.ndarray((nphi, ntheta, 4, 3)) # 4 faces associated with each vertex, defined by 3 vertices
        for i in range(nphi):
            for j in range(ntheta):
                if i == 0:
                    south_pole = True
                else:
                    south_pole = False
                    il = i - 1

                if i == nphi - 1:
                    north_pole = True
                else:
                    north_pole = False
                    ir = i + 1

                if j == 0:
                    jl = -1
                else:
                    jl = j - 1

                if j == ntheta - 1:
                    jr = 0
                else:
                    jr = j + 1

                faces[i,j,0,0] = vertex_id[i ,jr]
                faces[i,j,0,1] = vertex_id[i ,j ]
                if north_pole:
                    faces[i,j,0,2] = -2
                else:
                    faces[i,j,0,2] = vertex_id[ir,j ]

                if south_pole:
                    faces[i,j,1,0] = -1
                else:
                    faces[i,j,1,0] = vertex_id[il,j ]
                faces[i,j,1,1] = vertex_id[i ,j ]
                faces[i,j,1,2] = vertex_id[i ,jr]

                if south_pole:
                    faces[i,j,2,0] = -1
                else:
                    faces[i,j,2,0] = vertex_id[il,j ]
                faces[i,j,2,1] = vertex_id[i ,jl]
                faces[i,j,2,2] = vertex_id[i ,j ]

                faces[i,j,3,0] = vertex_id[i ,j ]
                faces[i,j,3,1] = vertex_id[i ,jl]
                if north_pole:
                    faces[i,j,3,2] = -2
                else:
                    faces[i,j,3,2] = vertex_id[ir,j ]

        vertices = vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        vertices = self.polar2cart(vertices)
        poles = self.polar2cart(poles)

        model = mesh.Mesh(np.zeros(faces.shape[0], dtype = mesh.Mesh.dtype))

        for i, f in enumerate(faces):
            for j in range(3):
                if f[j] == -1:
                    model.vectors[i][j] = poles[0]
                elif f[j] == -2:
                    model.vectors[i][j] = poles[1]
                else:
                    model.vectors[i][j] = vertices[f[j],:]

        model.save(filename)

    @staticmethod
    def polar2cart(rpt):
        n = rpt.shape[0]
        xyz = np.zeros((n,3))

        xyz[:,0] = rpt[:,0] * np.cos(rpt[:,2]) * np.sin(rpt[:,1])
        xyz[:,1] = rpt[:,0] * np.sin(rpt[:,2]) * np.sin(rpt[:,1])
        xyz[:,2] = rpt[:,0] * np.cos(rpt[:,1])

        return xyz
