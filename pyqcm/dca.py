import numpy as np
import pyqcm


class DCA_cluster:
    #----------------------------------------------------------------------
    def __init__(self, name, sites, superlattice, gen=None, lattice=None, basis=None):
        """
        :param str name: name of the cluster
        :param [[int]] sites: list of sites of the cluster, integer components
        :param [[int]] superlattice: superlattice basis vectors (integer components, the number of vectors is the dimension)
        :param [[int]] gen: generators of translation (one per dimension)
        :param [[float]]lattice: lattice basis vectors (integer components, optional). 
        :param [[float]] basis: basis vectors (real components). Optional. 

        """
        self.name = name
        self.sites = np.array(sites, dtype='int')
        self.superlattice = np.array(superlattice, dtype='int')
        if gen != None:
            self.gen = np.array(gen, dtype='int')
        else:
            self.gen = None
        self.dim = self.superlattice.shape[0]
        assert (self.dim > 0 and self.dim <= 3), 'DCA : the dimension must be 1, 2 or 3'

        # sanity check on the generators
        


        if lattice is None:  # default value for lattice
            self.lattice = np.eye(3, dtype='int')[0:self.dim,:]
        else:
            self.lattice = np.array(lattice)
            assert(self.lattice.shape[1] == 3 and self.lattice.shape[0] == self.dim), 'The lattice vectors have the wrong dimension or number'

        if basis is None:   # default value for basis
            self.basis = np.eye(3)[0:self.dim,:]
        else:
            self.basis = np.array(basis)
            assert(self.basis.shape[1] == 3 and self.basis.shape[0] == self.dim), 'The physical basis or its vectors have the wrong dimension'
            assert np.dot(self.basis[0],self.basis[1]) < -1e-9, 'Please choose a basis with an obtuse angle'

        # S : superlattice vectors in terms of the lattice vectors
        S = np.eye(3, dtype='int')
        S[0:self.dim, :] = self.superlattice
        latt = np.eye(3)
        latt[0:self.dim, :] = self.lattice
        ilatt = np.linalg.inv(latt)
        print('lattice :\n', latt, '\n', ilatt)
        S = np.dot(S,ilatt.T)

        # Sd : basis vectors of the reciprocal superlattice
        self.vol = int(np.rint(np.linalg.det(S)))
        print('volume of the supercell (in terms of the unit cell):', self.vol)
        Sd = np.eye(3, dtype='int')
        Sd[0,:] = np.cross(S[1,:], S[2,:])
        Sd[1,:] = np.cross(S[2,:], S[0,:])
        Sd[2,:] = np.cross(S[0,:], S[1,:])

        # Q = wavevectors of the reciprocal superlattice within the original Brillouin zone (2D for the moment)
        Q = []
        for i in range(-self.vol, self.vol):
            for j in range(-self.vol, self.vol):
                q = i*Sd[0] + j*Sd[1]
                if (q[0] >= self.vol)  or (q[1] >= self.vol) or (q[0] < 0)  or (q[1] < 0) :
                    continue
                Q.append(q)
        print('construction of ', len(Q), " wavevectors:")
        self.Q = np.array(Q, dtype='float')
        self.Q *= 2/self.vol
        print("wavevectors of the reciprocal superlattice within the original Brillouin zone (x pi):\n", self.Q)

        # creating the cluster models


    #----------------------------------------------------------------------
    def draw_patches(self, center=True, lim=1.05):
        """
        :param boolean center: If True, centers the plot at (0,0)
        :param float lim: the plot goes from -lim to lim in each direction (lim = 1 is pi)
        """
        import matplotlib.pyplot as plt
        from scipy.spatial import Voronoi, voronoi_plot_2d
        # zone boundary
        if self.dim != 2:
            print('draw_patches works only in dimension 2')
            return
        ZB = np.array([[0,0],[2,0],[2,2],[0,2],[0,0]])

        ibasis = np.linalg.inv(self.basis[0:2, 0:2])
        sites = np.dot(self.sites[:, 0:2],self.basis[0:2, 0:2])
        sup = np.dot(self.superlattice[0:2, 0:2],self.basis[0:2, 0:2])
        Qr = np.dot(self.Q[:, 0:2], ibasis.T)
        ZB = np.dot(ZB,ibasis.T)

        # basis vectors of the reciprocal lattice
        latt = np.eye(3)
        latt[0:self.dim, :] = self.lattice
        evol = int(np.rint(np.linalg.det(latt)))
        print('volume of the unit cell:', evol)
        K = np.eye(3) # basis vectors of the reciprocal lattice
        K[0,:] = np.cross(latt[1,:], latt[2,:])
        K[1,:] = np.cross(latt[2,:], latt[0,:])
        K[2,:] = np.cross(latt[0,:], latt[1,:])
        K *= 2/evol
        K = K[:self.dim,:self.dim]
        print('basis of reciprocal space:')
        for i in range(self.dim):
            print(K[i])
        K = np.dot(K,ibasis.T)

        if center:
            a = 2
            F = np.array([[a,0],[0,a],[-a,a],[-a,0],[0,-a],[a,-a],[a,0]])
            F = np.dot(F,ibasis.T)
            Fc = np.zeros(F.shape)
            norms = np.array([0.25*np.dot(F[i],F[i]) for i in range(F.shape[0])])
            for i in range(F.shape[0]-1):
                Fc[i] = np.linalg.solve(0.5*F[[i,i+1],:], norms[i:i+2])
            Fc[F.shape[0]-1] = Fc[0]
            ZB = Fc
            print(ZB)
            for i in range(self.vol):
                q = Qr[i]
                for p in F:
                    if np.linalg.norm(q-p) < np.linalg.norm(q):
                        Qr[i,:] = q-p

        # plot in direct space
        ms = 8
        plt.subplot(1,2,1)
        plt.gca().set_aspect(1)
        plt.grid()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(sites[:,0], sites[:,1], 'ko', ms=ms)
        plt.arrow(0, 0, sup[0,0], sup[0,1], width=0.01, head_width=0.1, length_includes_head=True, color='r')
        plt.arrow(0, 0, sup[1,0], sup[1,1], width=0.01, head_width=0.1, length_includes_head=True, color='r')
        plt.plot([sup[0,0]], [sup[0,1]], color='w')
        plt.plot([sup[1,0]], [sup[1,1]], color='w')


        # graphique dans l'espace réciproque
        plt.subplot(1,2,2)
        plt.gca().set_aspect(1)

        plt.plot(Qr[:,0], Qr[:,1], 'ko', ms=ms)
        plt.grid()
        plt.xlabel('$k_x/\pi$')
        plt.ylabel('$k_y/\pi$')

        # graphique de Voronoi
        # extension de la liste des points sur les zones voisines
        QrV = []
        for i,q in enumerate(Qr):
            QrV.append(q)
            QrV.append(q + K[0])
            QrV.append(q + K[1])
            QrV.append(q - K[0])
            QrV.append(q - K[1])
            QrV.append(q - K[0] + K[1])
            QrV.append(q + K[0] - K[1])
            QrV.append(q - K[0] - K[1])
            QrV.append(q + K[0] + K[1])
        QrV = np.array(QrV)

        vor = Voronoi(QrV)
        voronoi_plot_2d(vor, plt.gca())

        if center:
            plt.xlim(-lim,lim)
            plt.ylim(-lim,lim)
        plt.plot(ZB[:,0],ZB[:,1], 'b-')
        plt.tight_layout()
        plt.show()

    #----------------------------------------------------------------------

# def dca(DCA_cluster clus, N=N, grid='sharp'):




###########################################################################

##### 2x3 #####
sites = [
    ( 0, 0, 0),
    ( 1, 0, 0),
    ( 2, 0, 0),
    ( 0, 1, 0),
    ( 1, 1, 0),
    [2,1,0],
]
# superlattice=[
#     [3,-1, 0],
#     [3, 1, 0]
# ]
superlattice=[
    [3, 0, 0],
    [1, 2, 0]
]
basis = [( 1, 0, 0), [-0.5, 0.866025403784438, 0]]

X = DCA_cluster(name='G6', sites=sites, superlattice=superlattice, basis=None)
X.draw_patches()


##### G6 #####
# sites = [
#     ( 1, 0, 0),
#     ( 1, 1, 0),
#     ( 0, 1, 0),
#     (-1, 0, 0),
#     [-1,-1, 0],
#     [ 0,-1, 0],
# ]
# lattice = [[1, -1, 0], [2, 1, 0]]
# superlattice= [[3, 0, 0], [0, 3, 0]]
# basis = [( 1, 0, 0), [-0.5, 0.866025403784438, 0]]

# X = DCA_cluster(name='G6', sites=sites, superlattice=superlattice, lattice=lattice, basis=basis)
# X.draw_patches()


##### 9 sites triangular #####
# sites = [
#     [ 0, 0, 0],
#     ( 1, 0, 0),
#     ( 2, 0, 0),
#     ( 1, 1, 0),
#     [ 2, 1, 0],
#     [ 3, 1, 0],
#     [ 2, 2, 0],
#     [ 3, 2, 0],
#     [ 4, 2, 0]
# ]
# superlattice= [[3, 0, 0], [3, 3, 0]]
# basis = [( 1, 0, 0), [-0.5, 0.866025403784438, 0]]

# X = DCA_cluster(name='G6', sites=sites, superlattice=superlattice, basis=basis)
# X.draw_patches(lim=2)

##### T3 #####
# sites = [
#     [ 0,  0,  0], 
#     [ 1,  0,  0],
#     [ 1,  1,  0]     
# ]
# superlattice=[[1, -1, 0], [2, 1, 0]]
# basis = [( 1, 0, 0), [-0.5, 0.866025403784438, 0]]

# X = DCA_cluster(name='G6', sites=sites, superlattice=superlattice, basis=basis)
# X.draw_patches(lim=2)




##### T7 #####
# sites = [
#     ( 1, 0, 0),
#     ( 1, 1, 0),
#     ( 0, 1, 0),
#     (-1, 0, 0),
#     [-1, -1, 0],
#     [0, -1, 0],
#     [0, 0, 0]
# ]
# superlattice= [[2, -1, 0], [1, 3, 0]]
# basis = [( 1, 0, 0), [-0.5, 0.866025403784438, 0]]

# X = DCA_cluster(name='G6', sites=sites, superlattice=superlattice, basis=basis)
# X.draw_patches(center=True, lim=1.5)


