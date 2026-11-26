# Reaction families and reagents

The planner predicts a **family** (the broad class), then a **reagent** (one specific reaction inside that class). SMARTS spellings of the same reaction are apply-time fallbacks, not separate reagents.

Families apply to the learned planner only. The rule-based modifier ignores families; its source of truth is the flat reagent list in [reagents.md](reagents.md).

There are 12 families and 90 reagents. The sign column is the strategy-pool direction `(Δdensity, Δdetonation velocity, Δdetonation pressure, Δenthalpy of formation)`, each component in `{−1, 0, +1}`. It is a design label, not a measured delta. 78 reagents occur in the USPTO pairs. The other 12 stay so that every direction still has a reagent.

## Nitration

| Reagent | Edit | Direction |
| --- | --- | --- |
| `nitrate_arom_CH` | Aromatic C–H → C–NO2 | +1, +1, +1, 0 |
| `nitrate_alkyl_CH` | Aliphatic C–H → C–NO2 | +1, +1, +1, 0 |
| `nitramine_NH` | Amine N–H → nitramine N–NO2 | +1, +1, +1, +1 |
| `methyl_to_nitromethyl` | Ar–CH3 → Ar–CH2–NO2 | 0, +1, +1, 0 |
| `halogen_Br_to_nitro` | Ar–Br → Ar–NO2 | −1, +1, +1, +1 |
| `nitrate_ester_OH` | Alcohol O–H → nitrate ester O–NO2 | +1, +1, 0, 0 |
| `trinitromethyl` | Methyl → trinitromethyl C(NO2)3 | +1, +1, 0, +1 |
| `nitroso_arom` | Aromatic C–H → C–NO | 0, −1, +1, +1 |
| `iodo_to_nitro` | Aromatic C–I → C–NO2 | −1, +1, +1, −1 |
| `chloro_to_nitro` | Aromatic C–Cl → C–NO2 | −1, +1, +1, 0 |

Aromatic and aliphatic nitration share one direction. Nitramine and gem-dinitro share another. The parent atom (aromatic carbon, aliphatic carbon, N–H, or CH2) decides which of those can run.

## Azidation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `azide_arom` | Aromatic C–H → C–N3 | 0, 0, −1, +1 |
| `azide_methyl` | Ar–CH3 → Ar–CH2–N3 | 0, +1, −1, +1 |
| `amino_to_azide` | Ar–NH2 → Ar–N3 | −1, 0, 0, +1 |
| `chloro_to_azide` | Aromatic C–Cl → C–N3 | −1, +1, −1, +1 |
| `azide_alkyl` | Aliphatic C–H → C–N3 | +1, −1, +1, +1 |

## Heterocycle

| Reagent | Edit | Direction |
| --- | --- | --- |
| `tetrazole_CH` | Aromatic C–H → C–tetrazole | +1, 0, +1, +1 |
| `triazole_CH` | Aromatic C–H → C–(1,2,4-triazole) | 0, 0, +1, +1 |
| `pyrazole_CH` | Aromatic C–H → C–pyrazole | 0, 0, 0, +1 |
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
| `bromo_to_hydroxy` | Aromatic C–Br → C–OH | −1, +1, −1, 0 |
| `thiol_CH` | Aromatic C–H → C–SH | 0, +1, −1, −1 |
| `alcohol_to_ketone` | Primary alcohol CH2OH → aldehyde | 0, +1, 0, 0 |
| `carboxyl_alkyl` | Aliphatic C–H → C–COOH | +1, −1, −1, −1 |
| `hydroxyl_alkyl` | Aliphatic C–H → C–OH | +1, −1, −1, 0 |
| `formyl_alkyl` | Aliphatic C–H → C–CHO | +1, 0, 0, −1 |
| `methylsulfonyl_CH` | Aromatic C–H → C–SO2CH3 | +1, +1, −1, 0 |

## Amination

| Reagent | Edit | Direction |
| --- | --- | --- |
| `cyano_CH` | Aromatic C–H → C–CN | +1, 0, +1, +1 |
| `hydrazino_CH` | Aromatic C–H → C–NHNH2 | +1, 0, −1, +1 |
| `amino_CH` | Aromatic C–H → C–NH2 | 0, −1, −1, +1 |
| `amide_CH` | Aromatic C–H → C–CONH2 | 0, −1, +1, −1 |
| `isocyanate_CH` | Aromatic C–H → C–NCO | 0, +1, −1, 0 |
| `chloro_to_amino` | Aromatic C–Cl → C–NH2 | −1, −1, +1, +1 |
| `bromo_to_amino` | Aromatic C–Br → C–NH2 | −1, 0, +1, +1 |
| `iodo_to_cyano` | Aromatic C–I → C–CN | −1, +1, −1, −1 |
| `amino_alkyl` | Aliphatic C–H → C–NH2 | +1, −1, −1, +1 |
| `cyano_alkyl` | Aliphatic C–H → C–CN | +1, −1, 0, −1 |

## Halogenation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `fluoro_arom_CH` | Aromatic C–H → C–F | +1, +1, +1, −1 |
| `trifluoromethyl_CH` | Aromatic C–H → C–CF3 | +1, +1, 0, −1 |
| `chloro_arom_CH` | Aromatic C–H → C–Cl | +1, +1, −1, −1 |
| `bromo_arom_CH` | Aromatic C–H → C–Br | +1, +1, 0, 0 |
| `iodo_arom_CH` | Aromatic C–H → C–I | +1, 0, 0, 0 |
| `fluoro_alkyl` | Aliphatic C–H → C–F | +1, 0, +1, −1 |
| `iodo_to_fluoro` | Aromatic C–I → C–F | −1, +1, 0, −1 |
| `bromo_to_chloro` | Aromatic C–Br → C–Cl | −1, +1, 0, 0 |
| `bromo_to_fluoro` | Aromatic C–Br → C–F | −1, +1, 0, +1 |
| `chloro_alkyl` | Aliphatic C–H → C–Cl | +1, −1, 0, 0 |
| `bromo_alkyl` | Aliphatic C–H → C–Br | +1, −1, 0, +1 |
| `iodo_alkyl` | Aliphatic C–H → C–I | +1, −1, +1, 0 |

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
| `remove_phenyl` | Aromatic C–Ph → C–H | 0, 0, −1, 0 |

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
| `nitramine_to_nitrosamine` | Nitramine N–NO2 → nitrosamine N–NO | −1, −1, −1, +1 |
| `aliphatic_nitro_to_amino` | Aliphatic/cage C–NO2 → C–NH2 | 0, −1, −1, +1 |
| `remove_chloro_arom` | Aromatic C–Cl → C–H | −1, −1, 0, −1 |
| `remove_fluoro_arom` | Aromatic C–F → C–H | −1, −1, 0, +1 |
| `remove_iodo_arom` | Aromatic C–I → C–H | −1, −1, +1, −1 |
| `remove_trifluoromethyl` | Aromatic C–CF3 → C–H | −1, −1, +1, 0 |
| `remove_hydroxy_arom` | Aromatic C–OH → C–H | −1, 0, −1, −1 |
| `remove_methoxy_arom` | Aromatic C–OCH3 → C–H | −1, 0, 0, −1 |
| `remove_cyano_arom` | Aromatic C–CN → C–H | −1, 0, +1, 0 |

## Alkylation

| Reagent | Edit | Direction |
| --- | --- | --- |
| `methyl_CH` | Aromatic C–H → C–CH3 | −1, −1, 0, 0 |
| `ethyl_CH` | Aromatic C–H → C–CH2CH3 | −1, 0, −1, 0 |
| `hydroxymethyl_CH` | Aromatic C–H → C–CH2OH | 0, −1, 0, −1 |
| `acetyl_CH` | Aromatic C–H → C–COCH3 | 0, −1, −1, −1 |
| `tertbutyl_CH` | Aromatic C–H → C–C(CH3)3 | 0, −1, −1, 0 |
| `isopropyl_CH` | Aromatic C–H → C–CH(CH3)2 | 0, −1, 0, 0 |
| `vinyl_CH` | Aromatic C–H → C–CH=CH2 | 0, −1, 0, +1 |
| `allyl_CH` | Aromatic C–H → C–CH2CH=CH2 | +1, +1, −1, +1 |

## Aromatization

| Reagent | Edit | Direction |
| --- | --- | --- |
| `aromatize_diene` | Cyclohexadiene → benzene | 0, +1, 0, −1 |

## Identity

| Reagent | Edit | Direction |
| --- | --- | --- |
| `hold` | Carbon atom unchanged | 0, 0, 0, 0 |
