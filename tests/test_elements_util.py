from letalker.elements import _util as util
import numpy as np


def test_smb_vt_area_fun():
    # fmt:off
    #         Ap,    npc,   xc,    Ac,    nca,   Aa,    nain,  xin,    Ain,   xlip,   Alip,  xlar,  Alar,  nlarp, xp,    xa
    # params = [5.755, 7.363, 8.477, 0.951, 0.852, 6.655, 0.543, 17.028, 2.706, 17.099, 2.787, 1.866, 1.153, 0.418, 2.893, 14.385 ]
    # params = [5.755, 1.000, 8.477, 0.951, 1.000, 6.655, 1.000, 17.028, 2.706, 17.099, 2.787, 1.866, 1.153, 1.000, 2.893, 14.385 ]
    params = [5.755, 7.363, 8.477, 0.951, 0.852, 6.655, 0.543, 17.028, 2.706, 17.099, 2.787]
    # fmt:on

    func = util.smb_vt_area_fun(*params)
    x = np.linspace(-1.0, 18.0, 1001)
    f = func(x)
    assert x.shape == f.shape
