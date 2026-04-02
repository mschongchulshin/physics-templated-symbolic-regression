from .arrhenius         import fit as fit_arrhenius
from .thermal_softening import fit as fit_thermal_softening
from .calphad           import fit as fit_calphad
from .additive          import fit as fit_additive
from .multiplicative    import fit as fit_multiplicative
from .power_law         import fit as fit_power_law
from .entropy_weighted  import fit as fit_entropy_weighted
from .free_twostage     import fit as fit_free_twostage
from .free_singlestage  import fit as fit_free_singlestage

TEMPLATES = {
    "arrhenius":         fit_arrhenius,
    "thermal_softening": fit_thermal_softening,
    "calphad":           fit_calphad,
    "additive":          fit_additive,
    "multiplicative":    fit_multiplicative,
    "power_law":         fit_power_law,
    "entropy_weighted":  fit_entropy_weighted,
    "free_2stage":       fit_free_twostage,
    "free_1stage":       fit_free_singlestage,
}
