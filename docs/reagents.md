# Reagents (rule-based modifier)

The rule-based modifier works only at the reagent level. It is a flat list of reagents; reaction families play no part. Each reagent carries its effect on the four target properties. At run time IGNITE compares the seed's predicted properties with the targets, ranks reagents whose effects close the gap, and picks the reaction site at random (seeded) among that reagent's valid sites.

Effects were taken from the Direction column of [reaction_families.md](reaction_families.md); the reaction SMARTS live in `backend/app/services/reaction_primitives.py` under the same reagent id. The code list is `backend/app/services/reagents.py`; a unit test keeps this table, that list, and the primitive deltas in sync.

90 reagents. 78 of them occur in the USPTO pairs. The other 12 are kept so every direction in `{−1, 0, +1}^4` still has a reagent: those directions had no recorded patent hit. Some directions have more than one reagent, because those reactions are different even though their design signs match.

| Reagent | Edit | density | det_velocity | det_pressure | hf_solid |
| --- | --- | --- | --- | --- | --- |
| `nitrate_arom_CH` | Aromatic C–H → C–NO2 | raise | raise | raise | neutral |
| `nitrate_alkyl_CH` | Aliphatic C–H → C–NO2 | raise | raise | raise | neutral |
| `nitramine_NH` | Amine N–H → nitramine N–NO2 | raise | raise | raise | raise |
| `methyl_to_nitromethyl` | Ar–CH3 → Ar–CH2–NO2 | neutral | raise | raise | neutral |
| `halogen_Br_to_nitro` | Ar–Br → Ar–NO2 | lower | raise | raise | raise |
| `nitrate_ester_OH` | Alcohol O–H → nitrate ester O–NO2 | raise | raise | neutral | neutral |
| `trinitromethyl` | Methyl → trinitromethyl C(NO2)3 | raise | raise | neutral | raise |
| `nitroso_arom` | Aromatic C–H → C–NO | neutral | lower | raise | raise |
| `azide_arom` | Aromatic C–H → C–N3 | neutral | neutral | lower | raise |
| `azide_methyl` | Ar–CH3 → Ar–CH2–N3 | neutral | raise | lower | raise |
| `amino_to_azide` | Ar–NH2 → Ar–N3 | lower | neutral | neutral | raise |
| `tetrazole_CH` | Aromatic C–H → C–tetrazole | raise | neutral | raise | raise |
| `triazole_CH` | Aromatic C–H → C–(1,2,4-triazole) | neutral | neutral | raise | raise |
| `pyrazole_CH` | Aromatic C–H → C–pyrazole | neutral | neutral | neutral | raise |
| `nitrile_to_tetrazole` | Nitrile → 5-substituted tetrazole | raise | raise | raise | neutral |
| `n_oxide` | Aromatic N → N-oxide | neutral | raise | raise | raise |
| `hydroxyl_CH` | Aromatic C–H → C–OH | raise | lower | raise | lower |
| `methoxy_CH` | Aromatic C–H → C–OCH3 | raise | neutral | lower | neutral |
| `carboxyl_CH` | Aromatic C–H → C–COOH | neutral | neutral | raise | lower |
| `methylester_CH` | Aromatic C–H → C–COOCH3 | raise | neutral | lower | lower |
| `amino_to_nitro` | Ar–NH2 → Ar–NO2 | neutral | raise | raise | lower |
| `hydroperoxy_CH` | Aromatic C–H → C–OOH | raise | neutral | neutral | raise |
| `epoxidize_CC` | Aliphatic C=C → epoxide | raise | neutral | raise | neutral |
| `formyl_CH` | Aromatic C–H → C–CHO | neutral | lower | neutral | lower |
| `cyano_CH` | Aromatic C–H → C–CN | raise | neutral | raise | raise |
| `hydrazino_CH` | Aromatic C–H → C–NHNH2 | raise | neutral | lower | raise |
| `amino_CH` | Aromatic C–H → C–NH2 | neutral | lower | lower | raise |
| `amide_CH` | Aromatic C–H → C–CONH2 | neutral | lower | raise | lower |
| `isocyanate_CH` | Aromatic C–H → C–NCO | neutral | raise | lower | neutral |
| `fluoro_arom_CH` | Aromatic C–H → C–F | raise | raise | raise | lower |
| `trifluoromethyl_CH` | Aromatic C–H → C–CF3 | raise | raise | neutral | lower |
| `chloro_arom_CH` | Aromatic C–H → C–Cl | raise | raise | lower | lower |
| `bromo_arom_CH` | Aromatic C–H → C–Br | raise | raise | neutral | neutral |
| `iodo_arom_CH` | Aromatic C–H → C–I | raise | neutral | neutral | neutral |
| `fluoro_alkyl` | Aliphatic C–H → C–F | raise | neutral | raise | lower |
| `dehydrogenate_CC` | C(sp3)–C(sp3) → C=C | neutral | raise | lower | raise |
| `alkyne_CC` | Acyclic C–C → C≡C | neutral | raise | neutral | raise |
| `add_phenyl_CH` | Aromatic C–H → C–Ph | neutral | neutral | raise | neutral |
| `benzyl_CH` | Aromatic C–H → C–CH2Ph | neutral | lower | raise | neutral |
| `remove_nitro_arom` | Remove aromatic nitro | lower | lower | lower | neutral |
| `remove_bromo_arom` | Remove aromatic Br | lower | neutral | neutral | neutral |
| `remove_azide_arom` | Remove aromatic azide | neutral | neutral | neutral | lower |
| `nitro_to_methyl_arom` | Ar–NO2 → Ar–CH3 | lower | lower | lower | lower |
| `nitramine_to_amine` | Nitramine N–NO2 → amine N–H | neutral | neutral | lower | lower |
| `methyl_to_H` | Ar–CH3 → Ar–H | lower | neutral | lower | raise |
| `hydrogenate_CC` | Aliphatic C=C → C–C | neutral | lower | raise | lower |
| `nitro_to_amino` | Ar–NO2 → Ar–NH2 | neutral | lower | lower | raise |
| `azide_to_amino` | Ar–N3 → Ar–NH2 | lower | neutral | raise | lower |
| `methyl_CH` | Aromatic C–H → C–CH3 | lower | lower | neutral | neutral |
| `ethyl_CH` | Aromatic C–H → C–CH2CH3 | lower | neutral | lower | neutral |
| `hydroxymethyl_CH` | Aromatic C–H → C–CH2OH | neutral | lower | neutral | lower |
| `aromatize_diene` | Cyclohexadiene → benzene | neutral | raise | neutral | lower |
| `nitramine_to_nitrosamine` | Nitramine N–NO2 → nitrosamine N–NO | lower | lower | lower | raise |
| `aliphatic_nitro_to_amino` | Aliphatic/cage C–NO2 → C–NH2 | neutral | lower | lower | raise |
| `hold` | Carbon atom unchanged | neutral | neutral | neutral | neutral |
| `remove_chloro_arom` | Aromatic C–Cl → C–H | lower | lower | neutral | lower |
| `remove_fluoro_arom` | Aromatic C–F → C–H | lower | lower | neutral | raise |
| `remove_iodo_arom` | Aromatic C–I → C–H | lower | lower | raise | lower |
| `remove_trifluoromethyl` | Aromatic C–CF3 → C–H | lower | lower | raise | neutral |
| `chloro_to_amino` | Aromatic C–Cl → C–NH2 | lower | lower | raise | raise |
| `remove_hydroxy_arom` | Aromatic C–OH → C–H | lower | neutral | lower | lower |
| `remove_methoxy_arom` | Aromatic C–OCH3 → C–H | lower | neutral | neutral | lower |
| `remove_cyano_arom` | Aromatic C–CN → C–H | lower | neutral | raise | neutral |
| `bromo_to_amino` | Aromatic C–Br → C–NH2 | lower | neutral | raise | raise |
| `iodo_to_cyano` | Aromatic C–I → C–CN | lower | raise | lower | lower |
| `bromo_to_hydroxy` | Aromatic C–Br → C–OH | lower | raise | lower | neutral |
| `chloro_to_azide` | Aromatic C–Cl → C–N3 | lower | raise | lower | raise |
| `iodo_to_fluoro` | Aromatic C–I → C–F | lower | raise | neutral | lower |
| `bromo_to_chloro` | Aromatic C–Br → C–Cl | lower | raise | neutral | neutral |
| `bromo_to_fluoro` | Aromatic C–Br → C–F | lower | raise | neutral | raise |
| `iodo_to_nitro` | Aromatic C–I → C–NO2 | lower | raise | raise | lower |
| `chloro_to_nitro` | Aromatic C–Cl → C–NO2 | lower | raise | raise | neutral |
| `acetyl_CH` | Aromatic C–H → C–COCH3 | neutral | lower | lower | lower |
| `tertbutyl_CH` | Aromatic C–H → C–C(CH3)3 | neutral | lower | lower | neutral |
| `isopropyl_CH` | Aromatic C–H → C–CH(CH3)2 | neutral | lower | neutral | neutral |
| `vinyl_CH` | Aromatic C–H → C–CH=CH2 | neutral | lower | neutral | raise |
| `remove_phenyl` | Aromatic C–Ph → C–H | neutral | neutral | lower | neutral |
| `thiol_CH` | Aromatic C–H → C–SH | neutral | raise | lower | lower |
| `alcohol_to_ketone` | Primary alcohol CH2OH → aldehyde | neutral | raise | neutral | neutral |
| `carboxyl_alkyl` | Aliphatic C–H → C–COOH | raise | lower | lower | lower |
| `hydroxyl_alkyl` | Aliphatic C–H → C–OH | raise | lower | lower | neutral |
| `amino_alkyl` | Aliphatic C–H → C–NH2 | raise | lower | lower | raise |
| `cyano_alkyl` | Aliphatic C–H → C–CN | raise | lower | neutral | lower |
| `chloro_alkyl` | Aliphatic C–H → C–Cl | raise | lower | neutral | neutral |
| `bromo_alkyl` | Aliphatic C–H → C–Br | raise | lower | neutral | raise |
| `iodo_alkyl` | Aliphatic C–H → C–I | raise | lower | raise | neutral |
| `azide_alkyl` | Aliphatic C–H → C–N3 | raise | lower | raise | raise |
| `formyl_alkyl` | Aliphatic C–H → C–CHO | raise | neutral | neutral | lower |
| `methylsulfonyl_CH` | Aromatic C–H → C–SO2CH3 | raise | raise | lower | neutral |
| `allyl_CH` | Aromatic C–H → C–CH2CH=CH2 | raise | raise | lower | raise |

## Cage and saturated-nitramine reagents

These two reagents give the rule-based modifier reaction sites on saturated nitramines and fully substituted cages (RDX, HMX, CL-20, TEX, octanitrocubane), which otherwise match no aromatic/open-chain C–H or N–H pattern:

- `nitramine_to_nitrosamine` — N–NO2 → N–NO on a ring nitramine nitrogen; nitramine ⇄ nitrosamine interconversion is the first step of nitramine thermal/photolytic decomposition (Zeman, Propellants Explos. Pyrotech. 32, 155 (2007)). Matches RDX, HMX, CL-20, TEX. Kept so the direction `(−1, −1, −1, +1)` still has a reagent; it has no USPTO row.
- `aliphatic_nitro_to_amino` — aliphatic/cage C–NO2 → C–NH2 reduction, the standard nitro-to-amine reduction applied at an sp3 carbon. Matches octanitrocubane.
