# Reaction families and reagents

The planner predicts a **family** (the broad class), then a **reagent** (one specific reaction inside that class). SMARTS spellings of the same reaction are apply-time fallbacks, not separate reagents.

There are 11 families and 62 reagents. The sign column is the strategy-pool direction `(Δdensity, Δdetonation velocity, Δdetonation pressure, Δenthalpy of formation)`, each component in `{−1, 0, +1}`. It is a design label, not a measured delta.

## Nitration

| Reagent | Edit | Direction |
| --- | --- | --- |
| `nitrate_arom_CH` | Aromatic C–H → C–NO2 | +1, +1, +1, 0 |
| `nitrate_alkyl_CH` | Aliphatic C–H → C–NO2 | +1, +1, +1, 0 |
| `nitramine_NH` | Amine N–H → nitramine N–NO2 | +1, +1, +1, +1 |
| `methyl_to_nitromethyl` | Ar–CH3 → Ar–CH2–NO2 | 0, +1, +1, 0 |
| `halogen_Br_to_nitro` | Ar–Br → Ar–NO2 | −1, +1, +1, +1 |
| `nitrate_ester_OH` | Alcohol O–H → nitrate ester O–NO2 | +1, +1, 0, 0 |
| `gem_dinitro` | Aliphatic CH2 → gem-dinitro C(NO2)2 | +1, +1, +1, +1 |
| `trinitromethyl` | Methyl → trinitromethyl C(NO2)3 | +1, +1, 0, +1 |
| `nitroso_arom` | Aromatic C–H → C–NO | 0, −1, +1, +1 |

Aromatic and aliphatic nitration share one direction. Nitramine and gem-dinitro share another. The parent atom (aromatic carbon, aliphatic carbon, N–H, or CH2) decides which of those can run.

## Azidation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `azide_arom` | Aromatic C–H → C–N3 | 0, 0, −1, +1 |
| `azide_methyl` | Ar–CH3 → Ar–CH2–N3 | 0, +1, −1, +1 |
| `amino_to_azide` | Ar–NH2 → Ar–N3 | −1, 0, 0, +1 |
| `halogen_I_to_azide` | Ar–I → Ar–N3 | −1, +1, +1, +1 |

## Heterocycle

| Reagent | Edit | Direction |
| --- | --- | --- |
| `tetrazole_CH` | Aromatic C–H → C–tetrazole | +1, 0, +1, +1 |
| `triazole_CH` | Aromatic C–H → C–(1,2,4-triazole) | 0, 0, +1, +1 |
| `furazan_CH` | Aromatic C–H → C–furazan (1,2,5-oxadiazole) | +1, +1, +1, +1 |
| `furoxan_CH` | Aromatic C–H → C–furoxan | +1, +1, 0, +1 |
| `tetrazine_CH` | Aromatic C–H → C–(1,2,4,5-tetrazine) | 0, +1, +1, +1 |
| `oxadiazole_CH` | Aromatic C–H → C–(1,3,4-oxadiazole) | +1, 0, 0, +1 |
| `pyrazole_CH` | Aromatic C–H → C–pyrazole | 0, 0, 0, +1 |
| `imidazole_CH` | Aromatic C–H → C–imidazole | 0, 0, +1, 0 |
| `triazine_CH` | Aromatic C–H → C–(1,3,5-triazine) | 0, +1, 0, +1 |
| `nitrile_to_tetrazole` | Nitrile → 5-substituted tetrazole | +1, +1, +1, 0 |

## Oxidation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `n_oxide` | Aromatic N → N-oxide | 0, +1, +1, +1 |
| `hydroxyl_CH` | Aromatic C–H → C–OH | +1, −1, +1, −1 |
| `methoxy_CH` | Aromatic C–H → C–OCH3 | +1, 0, −1, 0 |
| `carboxyl_CH` | Aromatic C–H → C–COOH | 0, 0, +1, −1 |
| `methylester_CH` | Aromatic C–H → C–COOCH3 | +1, 0, −1, −1 |
| `amino_to_nitro` | Ar–NH2 → Ar–NO2 | 0, +1, +1, −1 |
| `hydroperoxy_CH` | Aromatic C–H → C–OOH | +1, 0, 0, +1 |
| `epoxidize_CC` | Aliphatic C=C → epoxide | +1, 0, +1, 0 |
| `formyl_CH` | Aromatic C–H → C–CHO | 0, −1, 0, −1 |

## Amination

| Reagent | Edit | Direction |
| --- | --- | --- |
| `cyano_CH` | Aromatic C–H → C–CN | +1, 0, +1, +1 |
| `hydrazino_CH` | Aromatic C–H → C–NHNH2 | +1, 0, −1, +1 |
| `amino_CH` | Aromatic C–H → C–NH2 | 0, −1, −1, +1 |
| `guanidino_CH` | Aromatic C–H → C–guanidine | 0, +1, 0, +1 |
| `amide_CH` | Aromatic C–H → C–CONH2 | 0, −1, +1, −1 |
| `isocyanate_CH` | Aromatic C–H → C–NCO | 0, +1, −1, 0 |

## Halogenation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `fluoro_arom_CH` | Aromatic C–H → C–F | +1, +1, +1, −1 |
| `trifluoromethyl_CH` | Aromatic C–H → C–CF3 | +1, +1, 0, −1 |
| `chloro_arom_CH` | Aromatic C–H → C–Cl | +1, +1, −1, −1 |
| `bromo_arom_CH` | Aromatic C–H → C–Br | +1, +1, 0, 0 |
| `iodo_arom_CH` | Aromatic C–H → C–I | +1, 0, 0, 0 |
| `difluoramino_CH` | Aromatic C–H → C–NF2 | +1, +1, +1, 0 |
| `fluoro_alkyl` | Aliphatic C–H → C–F | +1, 0, +1, −1 |

## Dehydrogenation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `dehydrogenate_CC` | C(sp3)–C(sp3) → C=C | 0, +1, −1, +1 |
| `alkyne_CC` | Acyclic C–C → C≡C | 0, +1, 0, +1 |

## Arylation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `add_phenyl_CH` | Aromatic C–H → C–Ph | 0, 0, +1, 0 |
| `benzyl_CH` | Aromatic C–H → C–CH2Ph | 0, −1, +1, 0 |

## Reduction

| Reagent | Edit | Direction |
| --- | --- | --- |
| `remove_nitro_arom` | Remove aromatic nitro | −1, −1, −1, 0 |
| `remove_bromo_arom` | Remove aromatic Br | −1, 0, 0, 0 |
| `remove_azide_arom` | Remove aromatic azide | 0, 0, 0, −1 |
| `nitro_to_methyl_arom` | Ar–NO2 → Ar–CH3 | −1, −1, −1, −1 |
| `nitramine_to_amine` | Nitramine N–NO2 → amine N–H | 0, 0, −1, −1 |
| `methyl_to_H` | Ar–CH3 → Ar–H | −1, 0, −1, +1 |
| `hydrogenate_CC` | Aliphatic C=C → C–C | 0, −1, +1, −1 |
| `nitro_to_amino` | Ar–NO2 → Ar–NH2 | 0, −1, −1, +1 |
| `azide_to_amino` | Ar–N3 → Ar–NH2 | −1, 0, +1, −1 |

## Alkylation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `methyl_CH` | Aromatic C–H → C–CH3 | −1, −1, 0, 0 |
| `ethyl_CH` | Aromatic C–H → C–CH2CH3 | −1, 0, −1, 0 |
| `hydroxymethyl_CH` | Aromatic C–H → C–CH2OH | 0, −1, 0, −1 |

## Aromatization

| Reagent | Edit | Direction |
| --- | --- | --- |
| `aromatize_diene` | Cyclohexadiene → benzene | 0, +1, 0, −1 |
