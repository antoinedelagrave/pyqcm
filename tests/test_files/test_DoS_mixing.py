# Test file
# Goal : to test that plot_DoS() plots, and prints to file, the spin components that are
#        appropriate to the mixing state of the solution, for all the bands of the model.
#        The spin down DoS is shown only for mixings 2 (spin flip), 3 (full Nambu) and
#        4 (spin dependent). For mixing 0 it is identical to the spin up DoS and for
#        mixing 1 it is the spin up DoS at the opposite frequency (both checked below).
#--------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pyqcm

# parameters producing each of the five mixing states
CASES = (
    (0, 'R0:N4:S0', "U=1\nt=1\nmu=1"),
    (1, 'R0:S0',    "U=1\nt=1\nmu=1\nS=0.4"),
    (2, 'R0:N4',    "U=1\nt=1\nmu=1\nH=0.5\nHx=1"),
    (3, 'R0',       "U=1\nt=1\nmu=1\nS=0.4\nH=0.5\nHx=1"),
    (4, 'R0:N4:S0', "U=1\nt=1\nmu=1\nH=0.5"),
)
SPIN_DOWN = {0: False, 1: False, 2: True, 3: True, 4: True}  # expected behavior of plot_DoS
COLORS = ['b', 'g']
W = np.arange(-3.0, 3.01, 0.01)
ETA = 0.04


def define_model(nband):
    """1D chain of 4 sites. nband=2 puts two sites in the unit cell of the lattice."""
    pyqcm.reset_model()
    CM = pyqcm.cluster_model(4, name='clus')
    clus = pyqcm.cluster(CM, ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)))
    model = pyqcm.lattice_model('1D_L4', clus, ((4, 0, 0),), ((2, 0, 0),) if nband == 2 else None)
    model.interaction_operator('U')
    model.hopping_operator('t', (1, 0, 0), -1)
    model.anomalous_operator('S', (0, 0, 0), 1)
    model.density_wave('H', 'Z', (0, 0, 0))
    model.density_wave('Hx', 'X', (0, 0, 0))
    model.density_wave('cdw', 'N', (1, 0, 0))  # makes the two bands inequivalent
    return model


def check(nband, mix, sector, params, labels, ax):
    """computes the DoS in a given mixing state and checks the output of plot_DoS()"""

    pyqcm.banner('plot_DoS with mixing = {:d}, {:d} band(s)'.format(mix, nband), '=')
    M = define_model(nband)
    M.set_target_sectors([sector])
    M.set_parameters(params + ("\ncdw=0.5" if nband > 1 else ""))
    I = pyqcm.model_instance(M)
    assert M.mixing == mix, 'expected mixing {:d}, got {:d}'.format(mix, M.mixing)
    assert M.nband == nband, 'expected {:d} band(s), got {:d}'.format(nband, M.nband)

    spin_down = SPIN_DOWN[mix]
    ncol = 2*nband if spin_down else nband
    data_file = 'dos_mix{:d}_{:d}band.tsv'.format(mix, nband)
    w, A = I.plot_DoS(W, eta=ETA, progress=False, labels=labels, colors=COLORS[0:nband], data_file=data_file, plt_ax=ax)

    # 1. the components returned depend on the mixing
    assert A.shape == (len(W), ncol), 'expected a DoS array of shape {}, got {}'.format((len(W), ncol), A.shape)

    # 2. the columns printed to file follow the same rule
    names = ['up_{:d}'.format(i+1) for i in range(nband)]
    if spin_down:
        names += ['down_{:d}'.format(i+1) for i in range(nband)]
    names += ['cumul_up_{:d}'.format(i+1) for i in range(nband)]
    if spin_down:
        names += ['cumul_down_{:d}'.format(i+1) for i in range(nband)]
    with open(data_file, 'r') as F:
        head = F.readline().split()
    assert head == ['w'] + names, 'expected the columns {}, got {}'.format(['w'] + names, head)
    D = np.genfromtxt(data_file, skip_header=1, max_rows=len(W))
    assert D.shape == (len(W), 1 + 2*ncol), 'expected data of shape {}, got {}'.format((len(W), 1 + 2*ncol), D.shape)
    assert np.allclose(D[:, 0], np.real(w)), 'the frequencies printed to file differ from those computed'
    assert np.allclose(D[:, 1:ncol+1], A, atol=1e-6), 'the DoS printed to file differs from the one returned'
    assert np.allclose(D[:, ncol+1:], np.cumsum(A, axis=0)*(W[1]-W[0]) - A[0, :]*(W[1]-W[0]), atol=1e-6), 'wrong cumulative DoS'

    # 3. one curve per band and per plotted spin, with the expected labels, colors and styles
    L = labels if labels is not None else ['orb {:d}'.format(i+1) for i in range(nband)]
    expected = []
    for i in range(nband):
        expected.append(L[i] + '$\\uparrow$' if spin_down else L[i])
        if spin_down:
            expected.append(L[i] + '$\\downarrow$')
    handles, curves = ax.get_legend_handles_labels()
    assert curves == expected, 'expected the curves {}, got {}'.format(expected, curves)
    for i in range(nband):
        up = handles[(2 if spin_down else 1)*i]
        assert to_rgba(up.get_color()) == to_rgba(COLORS[i]), 'band {:d} does not have the requested color'.format(i+1)
        if spin_down:  # the spin down curve of a band repeats its color, with a dashed line
            down = handles[2*i+1]
            assert to_rgba(down.get_color()) == to_rgba(COLORS[i]), 'the two spins of band {:d} differ in color'.format(i+1)
            assert down.get_linestyle() == '--', 'the spin down curve of band {:d} is not dashed'.format(i+1)

    # 4. all the bands are computed, and they are distinct here (see the 'cdw' parameter)
    if nband > 1:
        assert not np.allclose(A[:, 0], A[:, 1]), 'the DoS of the different bands should differ'

    # 5. why the spin down part is plotted, or not
    z = 0.6 + ETA*1j
    d = np.array(I.dos(z))
    if mix == 0:
        assert np.allclose(d[nband:], d[0:nband]), 'the spin down DoS should be identical to the spin up one'
    elif mix == 1:
        d_opp = np.array(I.dos(-z.real + ETA*1j))
        assert np.allclose(d[nband:], d_opp[0:nband]), 'the Nambu component should be the spin down DoS at the opposite frequency'
    else:
        assert not np.allclose(A[:, 0:nband], A[:, nband:]), 'the spin down DoS should differ from the spin up one'

    ax.set_title('mixing = {:d}'.format(mix), fontsize=9)
    ax.legend(fontsize=6)


#--------------------------------------------------------------------------------
# one band, with the default labels ('orb 1', 'orb 2', ...), then two bands with explicit labels

for nband, labels in ((1, None), (2, ['s', 'p'])):
    fig, axes = plt.subplots(1, len(CASES), figsize=(4*len(CASES), 3))
    for mix, sector, params in CASES:
        check(nband, mix, sector, params, labels, axes[mix])
    fig.suptitle('plot_DoS(), {:d} band(s)'.format(nband))
    fig.tight_layout()
    fig.savefig('test_DoS_mixing_{:d}band.pdf'.format(nband))
    plt.close(fig)

# the default labels number all the bands of a multiband model
fig, ax = plt.subplots()
check(2, 0, CASES[0][1], CASES[0][2], None, ax)
plt.close(fig)

# too few labels for the number of bands is an error
M = define_model(2)
M.set_target_sectors([CASES[0][1]])
M.set_parameters(CASES[0][2])
I = pyqcm.model_instance(M)
try:
    I.plot_DoS(W[0:4], eta=ETA, progress=False, labels=['only one'], data_file='dos_bad_labels.tsv', plt_ax=plt.gca())
except ValueError:
    pass
else:
    raise AssertionError('plot_DoS() should reject a list of labels shorter than the number of bands')
plt.close('all')

print('test_DoS_mixing : all checks passed')
