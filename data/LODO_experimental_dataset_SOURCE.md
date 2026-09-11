# Third-party data

`LODO_experimental_dataset.csv` is not ours. It is redistributed here so the
cross-source folds can be rebuilt from a clone.

## Source

> Borg, C. K. H., Frey, C., Moh, J., Pollock, T. M., Gorsse, S., Miracle, D. B.,
> Senkov, O. N., Meredig, B. & Saal, J. E. Expanded dataset of mechanical
> properties and observed phases of multi-principal element alloys.
> *Scientific Data* **7**, 430 (2020). https://doi.org/10.1038/s41597-020-00768-9

Deposit: https://figshare.com/articles/dataset/12642953

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0), which "permits use,
sharing, adaptation, distribution and reproduction in any medium or format, as
long as you give appropriate credit to the original author(s) and the source."

Redistribution here is under that licence, with the credit above.

## Modification

None. The file is byte-for-byte as deposited. Selection of the evaluable folds
happens in `lodo/build_folds.py`, which filters on composition family,
processing route and held-out source; nothing is edited in the table.
