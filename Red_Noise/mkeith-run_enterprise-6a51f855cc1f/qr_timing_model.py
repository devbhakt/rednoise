import numpy as np
from enterprise.signals.parameter import function
import enterprise.signals.utils as utils
from enterprise.signals.gp_signals import BasisGP

@function
def get_qr_basis(Mmat):
    q,r = np.linalg.qr(Mmat,mode='reduced')
    return q, np.ones(Mmat.shape[1])

def QRTimingModel(coefficients=False, name="linear_timing_model"):

    basis = get_qr_basis()
    prior = utils.tm_prior()

    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_qr"

        if coefficients:
            def _get_coefficient_logprior(self, key, c, **params):
                # MV: probably better to avoid this altogether
                #     than to use 1e40 as in get_phi
                return 0

    return TimingModel
