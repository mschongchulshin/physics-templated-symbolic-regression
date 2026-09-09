# Third-party data

`MPEA_figshare_dataset.csv` is not our data. It is redistributed here so the
cross-source folds can be rebuilt from a clone.

> Borg, C. K. H., Frey, C., Moh, J., Pollock, T. M., Gorsse, S., Miracle, D. B.,
> Senkov, O. N., Meredig, B. & Saal, J. E. Expanded dataset of mechanical
> properties and observed phases of multi-principal element alloys.
> *Scientific Data* **7**, 430 (2020). https://doi.org/10.1038/s41597-020-00768-9

Source: https://figshare.com/articles/dataset/12642953

The file is unmodified. Selection of the six evaluable folds happens in
`lodo/build_folds.py`, which filters on composition family, processing route
and held-out source; nothing is edited in the table itself.
