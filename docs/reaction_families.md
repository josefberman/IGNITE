# Reaction families and reagents

The planner predicts a **family** (the broad class), then a **reagent** (one specific reaction inside that class). SMARTS spellings of the same reaction are apply-time fallbacks, not separate reagents.

There are 11 families and 62 reagents. The sign column is the strategy-pool direction `(Δdensity, Δdetonation velocity, Δdetonation pressure, Δenthalpy of formation)`, each component in `{−1, 0, +1}`. It is a design label, not a measured delta. The Source column lists the literature cited for that reagent's three SMARTS variants and for the direction attached to them. "Group additivity" means the sign was extrapolated from Klapötke 2017 and Agrawal 2010. Full citations are at the end.

## Nitration

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `nitrate_arom_CH` | Aromatic C–H → C–NO2 | +1, +1, +1, 0 | Kamlet & Jacobs 1968; Klapötke 2017; Olah, Malhotra & Narang 1989; Zeman 2007 |
| `nitrate_alkyl_CH` | Aliphatic C–H → C–NO2 | +1, +1, +1, 0 | Klapötke 2017; Agrawal 2010; Zeman 2007 |
| `nitramine_NH` | Amine N–H → nitramine N–NO2 | +1, +1, +1, +1 | Klapötke 2017; Agrawal 2010; Kamlet & Jacobs 1968 |
| `methyl_to_nitromethyl` | Ar–CH3 → Ar–CH2–NO2 | 0, +1, +1, 0 | Korolev et al. 2006; Klapötke 2017; Agrawal 2010 |
| `halogen_Br_to_nitro` | Ar–Br → Ar–NO2 | −1, +1, +1, +1 | Klapötke 2017; Olah, Malhotra & Narang 1989; Zeman 2007 |
| `nitrate_ester_OH` | Alcohol O–H → nitrate ester O–NO2 | +1, +1, 0, 0 | Klapötke 2017; Agrawal 2010; Kamlet & Jacobs 1968 |
| `gem_dinitro` | Aliphatic CH2 → gem-dinitro C(NO2)2 | +1, +1, +1, +1 | Klapötke 2017; Agrawal 2010; Kamlet & Jacobs 1968 |
| `trinitromethyl` | Methyl → trinitromethyl C(NO2)3 | +1, +1, 0, +1 | Klapötke 2017; Agrawal 2010; Kamlet & Jacobs 1968 |
| `nitroso_arom` | Aromatic C–H → C–NO | 0, −1, +1, +1 | Group additivity; Agrawal 2010; Klapötke 2017 |

Aromatic and aliphatic nitration share one direction. Nitramine and gem-dinitro share another. The parent atom (aromatic carbon, aliphatic carbon, N–H, or CH2) decides which of those can run.

## Azidation

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `azide_arom` | Aromatic C–H → C–N3 | 0, 0, −1, +1 | Klapötke 2017; Bräse et al. 2005; Katritzky et al. 2010 |
| `azide_methyl` | Ar–CH3 → Ar–CH2–N3 | 0, +1, −1, +1 | Gao & Shreeve 2011; Bräse et al. 2005; Katritzky et al. 2010 |
| `amino_to_azide` | Ar–NH2 → Ar–N3 | −1, 0, 0, +1 | Bräse et al. 2005; Katritzky et al. 2010; Klapötke 2017 |
| `halogen_I_to_azide` | Ar–I → Ar–N3 | −1, +1, +1, +1 | Katritzky et al. 2010; Bräse et al. 2005; Klapötke 2017 |

## Heterocycle

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `tetrazole_CH` | Aromatic C–H → C–tetrazole | +1, 0, +1, +1 | Gao & Shreeve 2011; Klapötke 2017; Agrawal 2010 |
| `triazole_CH` | Aromatic C–H → C–(1,2,4-triazole) | 0, 0, +1, +1 | Gao & Shreeve 2011; Klapötke 2017; Agrawal 2010 |
| `furazan_CH` | Aromatic C–H → C–furazan (1,2,5-oxadiazole) | +1, +1, +1, +1 | Pagoria et al. 2002; Gao & Shreeve 2011; Klapötke 2017 |
| `furoxan_CH` | Aromatic C–H → C–furoxan | +1, +1, 0, +1 | Pagoria et al. 2002; Gao & Shreeve 2011; Klapötke 2017 |
| `tetrazine_CH` | Aromatic C–H → C–(1,2,4,5-tetrazine) | 0, +1, +1, +1 | Chavez et al. 2008; Gao & Shreeve 2011; Klapötke 2017 |
| `oxadiazole_CH` | Aromatic C–H → C–(1,3,4-oxadiazole) | +1, 0, 0, +1 | Gao & Shreeve 2011; Pagoria et al. 2002; Klapötke 2017 |
| `pyrazole_CH` | Aromatic C–H → C–pyrazole | 0, 0, 0, +1 | Katritzky et al. 2010; Gao & Shreeve 2011; Klapötke 2017 |
| `imidazole_CH` | Aromatic C–H → C–imidazole | 0, 0, +1, 0 | Katritzky et al. 2010; Gao & Shreeve 2011; Klapötke 2017 |
| `triazine_CH` | Aromatic C–H → C–(1,3,5-triazine) | 0, +1, 0, +1 | Coburn 1968; Pagoria et al. 2002; Klapötke 2017 |
| `nitrile_to_tetrazole` | Nitrile → 5-substituted tetrazole | +1, +1, +1, 0 | Klapötke 2017; Katritzky et al. 2010; Gao & Shreeve 2011 |

## Oxidation

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `n_oxide` | Aromatic N → N-oxide | 0, +1, +1, +1 | Pagoria et al. 2002; Klapötke 2017; Gao & Shreeve 2011 |
| `hydroxyl_CH` | Aromatic C–H → C–OH | +1, −1, +1, −1 | Zhang et al. 2007; Klapötke 2017; Politzer & Murray 2014 |
| `methoxy_CH` | Aromatic C–H → C–OCH3 | +1, 0, −1, 0 | Group additivity; Agrawal 2010; Klapötke 2017 |
| `carboxyl_CH` | Aromatic C–H → C–COOH | 0, 0, +1, −1 | Zhang et al. 2013; group additivity; Agrawal 2010 |
| `methylester_CH` | Aromatic C–H → C–COOCH3 | +1, 0, −1, −1 | Group additivity; Agrawal 2010; Klapötke 2017 |
| `amino_to_nitro` | Ar–NH2 → Ar–NO2 | 0, +1, +1, −1 | Klapötke 2017; Olah, Malhotra & Narang 1989; Zeman 2007 |
| `hydroperoxy_CH` | Aromatic C–H → C–OOH | +1, 0, 0, +1 | Group additivity; Politzer & Murray 2014; Klapötke 2017 |
| `epoxidize_CC` | Aliphatic C=C → epoxide | +1, 0, +1, 0 | Group additivity; Agrawal 2010; Klapötke 2017 |
| `formyl_CH` | Aromatic C–H → C–CHO | 0, −1, 0, −1 | Group additivity; Agrawal 2010; Zhang et al. 2013 |

## Amination

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `cyano_CH` | Aromatic C–H → C–CN | +1, 0, +1, +1 | Gao & Shreeve 2011; Klapötke 2017; Agrawal 2010 |
| `hydrazino_CH` | Aromatic C–H → C–NHNH2 | +1, 0, −1, +1 | Coburn 1968; Gao & Shreeve 2011; Klapötke 2017 |
| `amino_CH` | Aromatic C–H → C–NH2 | 0, −1, −1, +1 | Agrawal 2010; Klapötke 2017; Gao & Shreeve 2011 |
| `guanidino_CH` | Aromatic C–H → C–guanidine | 0, +1, 0, +1 | Klapötke 2017; Katritzky et al. 2010; Gao & Shreeve 2011 |
| `amide_CH` | Aromatic C–H → C–CONH2 | 0, −1, +1, −1 | Group additivity; Agrawal 2010; Zhang et al. 2013 |
| `isocyanate_CH` | Aromatic C–H → C–NCO | 0, +1, −1, 0 | Group additivity; Katritzky et al. 2010; Klapötke 2017 |

## Halogenation

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `fluoro_arom_CH` | Aromatic C–H → C–F | +1, +1, +1, −1 | Politzer & Murray 2014; Klapötke 2017; Agrawal 2010 |
| `trifluoromethyl_CH` | Aromatic C–H → C–CF3 | +1, +1, 0, −1 | Gao et al. 2013; Politzer & Murray 2014; Klapötke 2017 |
| `chloro_arom_CH` | Aromatic C–H → C–Cl | +1, +1, −1, −1 | Sikder et al. 2004; Politzer & Murray 2014; Klapötke 2017 |
| `bromo_arom_CH` | Aromatic C–H → C–Br | +1, +1, 0, 0 | Group additivity; Klapötke 2017; Agrawal 2010 |
| `iodo_arom_CH` | Aromatic C–H → C–I | +1, 0, 0, 0 | Politzer & Murray 2014; group additivity; Klapötke 2017 |
| `difluoramino_CH` | Aromatic C–H → C–NF2 | +1, +1, +1, 0 | Klapötke 2017; Agrawal 2010; Kamlet & Jacobs 1968 |
| `fluoro_alkyl` | Aliphatic C–H → C–F | +1, 0, +1, −1 | Klapötke 2017; Agrawal 2010; group additivity |

## Dehydrogenation

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `dehydrogenate_CC` | C(sp3)–C(sp3) → C=C | 0, +1, −1, +1 | Chavez et al. 2008; Klapötke 2017; Agrawal 2010 |
| `alkyne_CC` | Acyclic C–C → C≡C | 0, +1, 0, +1 | Group additivity; Klapötke 2017; Agrawal 2010 |

## Arylation

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `add_phenyl_CH` | Aromatic C–H → C–Ph | 0, 0, +1, 0 | Politzer et al. 2004; Politzer & Murray 2014; Klapötke 2017 |
| `benzyl_CH` | Aromatic C–H → C–CH2Ph | 0, −1, +1, 0 | Politzer et al. 2004; Politzer & Murray 2014; Klapötke 2017 |

## Reduction

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `remove_nitro_arom` | Remove aromatic nitro | −1, −1, −1, 0 | Group additivity; Kamlet & Jacobs 1968; Klapötke 2017; Zeman 2007 |
| `remove_bromo_arom` | Remove aromatic Br | −1, 0, 0, 0 | Group additivity; Sikder et al. 2004; Klapötke 2017 |
| `remove_azide_arom` | Remove aromatic azide | 0, 0, 0, −1 | Group additivity; Bräse et al. 2005; Katritzky et al. 2010 |
| `nitro_to_methyl_arom` | Ar–NO2 → Ar–CH3 | −1, −1, −1, −1 | Group additivity; Klapötke 2017; Agrawal 2010 |
| `nitramine_to_amine` | Nitramine N–NO2 → amine N–H | 0, 0, −1, −1 | Group additivity; Klapötke 2017; Agrawal 2010 |
| `methyl_to_H` | Ar–CH3 → Ar–H | −1, 0, −1, +1 | Group additivity; Klapötke 2017; Politzer & Murray 2014 |
| `hydrogenate_CC` | Aliphatic C=C → C–C | 0, −1, +1, −1 | Group additivity; Klapötke 2017; Agrawal 2010 |
| `nitro_to_amino` | Ar–NO2 → Ar–NH2 | 0, −1, −1, +1 | Olah, Malhotra & Narang 1989; Klapötke 2017; Zeman 2007 |
| `azide_to_amino` | Ar–N3 → Ar–NH2 | −1, 0, +1, −1 | Bräse et al. 2005; Katritzky et al. 2010; Klapötke 2017 |

## Alkylation

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `methyl_CH` | Aromatic C–H → C–CH3 | −1, −1, 0, 0 | Group additivity; Agrawal 2010; Klapötke 2017 |
| `ethyl_CH` | Aromatic C–H → C–CH2CH3 | −1, 0, −1, 0 | Group additivity; Agrawal 2010; Klapötke 2017 |
| `hydroxymethyl_CH` | Aromatic C–H → C–CH2OH | 0, −1, 0, −1 | Group additivity; Politzer & Murray 2014; Zhang et al. 2013 |

## Aromatization

| Reagent | Edit | Direction | Source |
| --- | --- | --- | --- |
| `aromatize_diene` | Cyclohexadiene → benzene | 0, +1, 0, −1 | Politzer & Murray 2014; Klapötke 2017; group additivity |

## References

- Agrawal 2010 — Agrawal, *High Energy Materials*, Wiley-VCH, 2010.
- Bräse et al. 2005 — Bräse et al., *Angew. Chem. Int. Ed.* 44, 5188 (2005).
- Chavez et al. 2008 — Chavez et al., *Angew. Chem. Int. Ed.* 47, 8307 (2008).
- Coburn 1968 — Coburn, *J. Heterocycl. Chem.* 5, 83 (1968).
- Gao & Shreeve 2011 — Gao & Shreeve, *Chem. Rev.* 2011, 111, 7377.
- Gao et al. 2013 — Gao et al., *RSC Adv.* 3, 4245 (2013).
- Group additivity — extrapolation from Klapötke 2017 and Agrawal 2010.
- Kamlet & Jacobs 1968 — Kamlet & Jacobs, *J. Chem. Phys.* 48, 23 (1968), cited together with Klapötke 2017.
- Katritzky et al. 2010 — Katritzky et al., *Chem. Rev.* 110, 2709 (2010).
- Klapötke 2017 — Klapötke, *Chemistry of High-Energy Materials*, 3rd ed., 2017.
- Korolev et al. 2006 — Korolev et al., *Russ. Chem. Bull.* 55, 1388 (2006).
- Olah, Malhotra & Narang 1989 — Olah, Malhotra & Narang, *Nitration: Methods and Mechanisms*, VCH/Wiley, 1989.
- Pagoria et al. 2002 — Pagoria et al., *Thermochim. Acta* 384, 187 (2002).
- Politzer & Murray 2014 — Politzer & Murray, *J. Mol. Model.* 2014, 20, 2223.
- Politzer et al. 2004 — Politzer et al., *J. Mol. Struct.* 684, 15 (2004).
- Sikder et al. 2004 — Sikder et al., *J. Hazard. Mater.* 112, 1 (2004).
- Zeman 2007 — Zeman, *Propellants Explos. Pyrotech.* 32, 155 (2007).
- Zhang et al. 2007 — Zhang et al., *J. Phys. Chem. B* 111, 14295 (2007).
- Zhang et al. 2013 — Zhang et al., *CrystEngComm* 15, 4003 (2013).
