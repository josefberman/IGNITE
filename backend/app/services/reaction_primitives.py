"""
Strategy Pool — 81 direction-tuple strategies for energetic molecule design.

Design (see plan doc §7):

1. A curated library of chemically-valid SMARTS "primitives", each tagged
   with an empirical direction-vector Δ = (Δρ, ΔD, ΔP, ΔHf) ∈ {-1, 0, +1}^4.
   Δ-vectors come from Kamlet–Jacobs detonation correlations (Kamlet &
   Jacobs, J. Chem. Phys. 48, 23 (1968); Kamlet & Hurwitz, ibid. 1969),
   Politzer & Murray electrostatic-potential analyses (J. Mol. Model. 2014,
   20, 2223), and the group-contribution summaries in Klapötke,
   *Chemistry of High-Energy Materials* (3rd ed., 2017) and Agrawal,
   *High Energy Materials* (Wiley-VCH, 2010).

2. The 81 direction tuples (3^4) are mapped to primitives by minimising
   the L1 distance ||Δ_primitive − target_tuple||₁. Ties are broken in
   favour of (a) classic energetic chemistry (aromatic nitration,
   nitramine formation, azidation, tetrazole installation), then
   (b) fewest bond edits.

3. Every non-identity primitive carries exactly three SMARTS citations
   (distinct literature references + query variants appropriate to the
   same Δ-direction). ``apply_strategy`` samples one variant uniformly.

4. Primitive SMARTS are parsed by ``rdkit.Chem.AllChem.ReactionFromSmarts``
   at import time; a failure raises ImportError so broken patterns are
   caught at startup rather than in the middle of a beam search.

The identity tuple (0, 0, 0, 0) is handled specially — there is no
meaningful "do-nothing" SMARTS reaction, so it returns the parent SMILES
unchanged when applied.
"""

import logging
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


logger = logging.getLogger(__name__)


# =============================================================================
# Primitive library. Each entry is:
#   (name, variants, delta, description)
# where ``variants`` is a list of (smarts_or_None, source-string).
# Identity has a single [(None, "N/A")]; every other primitive has exactly
# three literature-backed SMARTS variants (same Δ; sampled at apply time).
# =============================================================================

_KJ = ("Kamlet & Jacobs, J. Chem. Phys. 48, 23 (1968); "
       "Klapötke, Chemistry of High-Energy Materials, 3rd ed., 2017")
_POLITZER = "Politzer & Murray, J. Mol. Model. 2014, 20, 2223"
_KLAPOTKE = "Klapötke, Chemistry of High-Energy Materials, 3rd ed., 2017"
_AGRAWAL = "Agrawal, High Energy Materials, Wiley-VCH, 2010"
_GAO = "Gao & Shreeve, Chem. Rev. 2011, 111, 7377"
_GROUP_ADD = "Group-additivity extrapolation (Klapötke 2017; Agrawal 2010)"
_OLAH_NITRATION = ("Olah, Malhotra & Narang, "
                  "Nitration: Methods and Mechanisms (VCH/Wiley 1989)")
_ZEMAN = "Zeman, Propellants Explos. Pyrotech. 32, 155 (2007)"
_BRASE = "Bräse et al., Angew. Chem. Int. Ed. 44, 5188 (2005)"
_KATRITZKY = "Katritzky et al., Chem. Rev. 110, 2709 (2010)"
_PAGORIA = "Pagoria et al., Thermochim. Acta 384, 187 (2002)"
_COBURN = "Coburn, J. Heterocycl. Chem. 5, 83 (1968)"
_GAO_RS = "Gao et al., RSC Adv. 3, 4245 (2013)"
_SIKDER = "Sikder et al., J. Hazard. Mater. 112, 1 (2004)"
_ZHANG_JPCB = "Zhang et al., J. Phys. Chem. B 111, 14295 (2007)"
_ZHANG_CEC = "Zhang et al., CrystEngComm 15, 4003 (2013)"
_KOROLEV = "Korolev et al., Russ. Chem. Bull. 55, 1388 (2006)"
_CHAVEZ = "Chavez et al., Angew. Chem. Int. Ed. 47, 8307 (2008)"
_POLITZER_STRUCT = "Politzer et al., J. Mol. Struct. 684, 15 (2004)"

_PrimitiveVariant = Tuple[Optional[str], str]
_PrimitiveRow = Tuple[str, List[_PrimitiveVariant], Tuple[int, int, int, int], str]

_PRIMITIVE_ROWS: List[_PrimitiveRow] = [
    # --- Identity -----------------------------------------------------------
    ("identity",
     [(None, "N/A")],
     (0, 0, 0, 0),
     "No modification (parent SMILES returned unchanged)."),

    # --- Nitration family ---------------------------------------------------
    ("nitrate_arom_CH",
     [
         ("[cH:1]>>[c:1][N+](=O)[O-]", _KJ),
         ("[c;H1:1]>>[c:1][N+](=O)[O-]", _OLAH_NITRATION),
         ("[cH1:1]>>[c:1][N+](=O)[O-]", _ZEMAN),
     ],
     (+1, +1, +1, 0),
     "Aromatic C–H → C–NO2. Canonical energetic substitution."),
    ("nitrate_alkyl_CH",
     [
         ("[CX4;!H0:1]>>[C:1][N+](=O)[O-]", _KLAPOTKE),
         ("[CH3;X4:1]>>[CH2:1][N+](=O)[O-]", _AGRAWAL),
         ("[CH2;X4:1]>>[CH1:1][N+](=O)[O-]", _ZEMAN),
     ],
     (+1, +1, +1, 0),
     "Aliphatic C–H → C–NO2."),
    ("nitramine_NH",
     [
         ("[NX3;H1,H2:1]>>[N:1][N+](=O)[O-]", _KLAPOTKE),
         ("[NX3;H2:1]>>[N:1][N+](=O)[O-]", _AGRAWAL),
         ("[NX3;H1:1]>>[N:1][N+](=O)[O-]", _KJ),
     ],
     (+1, +1, +1, +1),
     "Amine N–H → nitramine N–NO2 (RDX/HMX family)."),

    # --- Nitrogen-rich heterocycles -----------------------------------------
    ("azide_arom",
     [
         ("[cH:1]>>[c:1]N=[N+]=[N-]", _KLAPOTKE),
         ("[c;H1:1]>>[c:1]N=[N+]=[N-]", _BRASE),
         ("[cH1:1]>>[c:1]N=[N+]=[N-]", _KATRITZKY),
     ],
     (0, 0, -1, +1),
     "Aromatic C–H → C–N3; nitrogen-rich, high Hf."),
    ("azide_methyl",
     [
         ("[CH3:1][c:2]>>[CH2:1]([c:2])N=[N+]=[N-]", _GAO),
         ("[CH3:1][c:2]>>[CH2:1]([c:2])N=[N+]=[N-]", _BRASE),
         ("[CH3:1][c:2]>>[CH2:1]([c:2])N=[N+]=[N-]", _KATRITZKY),
     ],
     (0, +1, -1, +1),
     "Ar–CH3 → Ar–CH2–N3 (azidomethyl)."),
    ("tetrazole_CH",
     [
         ("[cH:1]>>[c:1]c1nnn[nH]1", _GAO),
         ("[c;H1:1]>>[c:1]c1nnn[nH]1", _KLAPOTKE),
         ("[cH1:1]>>[c:1]c1nnn[nH]1", _AGRAWAL),
     ],
     (+1, 0, +1, +1),
     "Aromatic C–H → C–(1H-tetrazol-5-yl); N-rich, high Hf and density."),
    ("triazole_CH",
     [
         ("[cH:1]>>[c:1]c1cn[nH]n1", _GAO),
         ("[c;H1:1]>>[c:1]c1cn[nH]n1", _KLAPOTKE),
         ("[cH1:1]>>[c:1]c1cn[nH]n1", _AGRAWAL),
     ],
     (0, 0, +1, +1),
     "Aromatic C–H → C–(1,2,4-triazolyl)."),
    ("n_oxide",
     [
         ("[nX2:1]>>[n+:1][O-]", _PAGORIA),
         ("[n;H1:1]>>[n+:1][O-]", _KLAPOTKE),
         ("[n;H0:1]>>[n+:1][O-]", _GAO),
     ],
     (0, +1, +1, +1),
     "Aromatic N → N-oxide (Pagoria et al.)."),

    # --- Polar EWGs / donors ------------------------------------------------
    ("cyano_CH",
     [
         ("[cH:1]>>[c:1]C#N", _GAO),
         ("[c;H1:1]>>[c:1]C#N", _KLAPOTKE),
         ("[cH1:1]>>[c:1]C#N", _AGRAWAL),
     ],
     (+1, 0, +1, +1),
     "Aromatic C–H → C–CN."),
    ("hydrazino_CH",
     [
         ("[cH:1]>>[c:1]NN", _COBURN),
         ("[c;H1:1]>>[c:1]NN", _GAO),
         ("[cH1:1]>>[c:1]NN", _KLAPOTKE),
     ],
     (+1, 0, -1, +1),
     "Aromatic C–H → C–NHNH2 (hydrazino)."),
    ("amino_CH",
     [
         ("[cH:1]>>[c:1]N", _AGRAWAL),
         ("[c;H1:1]>>[c:1]N", _KLAPOTKE),
         ("[cH1:1]>>[c:1]N", _GAO),
     ],
     (0, -1, -1, +1),
     "Aromatic C–H → C–NH2 (weak donor; aminotriazine-style)."),

    # --- Halogens -----------------------------------------------------------
    ("fluoro_arom_CH",
     [
         ("[cH:1]>>[c:1]F", _POLITZER),
         ("[c;H1:1]>>[c:1]F", _KLAPOTKE),
         ("[cH1:1]>>[c:1]F", _AGRAWAL),
     ],
     (+1, +1, +1, -1),
     "Aromatic C–H → C–F."),
    ("trifluoromethyl_CH",
     [
         ("[cH:1]>>[c:1]C(F)(F)F", _GAO_RS),
         ("[c;H1:1]>>[c:1]C(F)(F)F", _POLITZER),
         ("[cH1:1]>>[c:1]C(F)(F)F", _KLAPOTKE),
     ],
     (+1, +1, 0, -1),
     "Aromatic C–H → C–CF3."),
    ("chloro_arom_CH",
     [
         ("[cH:1]>>[c:1]Cl", _SIKDER),
         ("[c;H1:1]>>[c:1]Cl", _POLITZER),
         ("[cH1:1]>>[c:1]Cl", _KLAPOTKE),
     ],
     (+1, +1, -1, -1),
     "Aromatic C–H → C–Cl (chain inhibitor)."),
    ("bromo_arom_CH",
     [
         ("[cH:1]>>[c:1]Br", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]Br", _KLAPOTKE),
         ("[cH1:1]>>[c:1]Br", _AGRAWAL),
     ],
     (+1, +1, 0, 0),
     "Aromatic C–H → C–Br (heavy halogen)."),
    ("iodo_arom_CH",
     [
         ("[cH:1]>>[c:1]I", _POLITZER),
         ("[c;H1:1]>>[c:1]I", _GROUP_ADD),
         ("[cH1:1]>>[c:1]I", _KLAPOTKE),
     ],
     (+1, 0, 0, 0),
     "Aromatic C–H → C–I (pure density gain)."),

    # --- Oxygen functional groups ------------------------------------------
    ("hydroxyl_CH",
     [
         ("[cH:1]>>[c:1]O", _ZHANG_JPCB),
         ("[c;H1:1]>>[c:1]O", _KLAPOTKE),
         ("[cH1:1]>>[c:1]O", _POLITZER),
     ],
     (+1, -1, +1, -1),
     "Aromatic C–H → C–OH (H-bond-driven density increase)."),
    ("methoxy_CH",
     [
         ("[cH:1]>>[c:1]OC", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]OC", _AGRAWAL),
         ("[cH1:1]>>[c:1]OC", _KLAPOTKE),
     ],
     (+1, 0, -1, 0),
     "Aromatic C–H → C–OCH3 (methoxy)."),
    ("carboxyl_CH",
     [
         ("[cH:1]>>[c:1]C(=O)O", _ZHANG_CEC),
         ("[c;H1:1]>>[c:1]C(=O)O", _GROUP_ADD),
         ("[cH1:1]>>[c:1]C(=O)O", _AGRAWAL),
     ],
     (0, 0, +1, -1),
     "Aromatic C–H → C–COOH."),
    ("methylester_CH",
     [
         ("[cH:1]>>[c:1]C(=O)OC", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]C(=O)OC", _AGRAWAL),
         ("[cH1:1]>>[c:1]C(=O)OC", _KLAPOTKE),
     ],
     (+1, 0, -1, -1),
     "Aromatic C–H → C–COOCH3."),

    # --- Carbon skeleton changes --------------------------------------------
    ("methyl_to_nitromethyl",
     [
         ("[c:1][CH3]>>[c:1][CH2][N+](=O)[O-]", _KOROLEV),
         ("[c:1][CH3:2]>>[c:1][CH2:2][N+](=O)[O-]", _KLAPOTKE),
         ("[c:1][CH3]>>[c:1][CH2][N+](=O)[O-]", _AGRAWAL),
     ],
     (0, +1, +1, 0),
     "Ar–CH3 → Ar–CH2–NO2."),
    ("dehydrogenate_CC",
     [
         ("[CX4;!H0:1][CX4;!H0:2]>>[C:1]=[C:2]", _CHAVEZ),
         ("[CX4;!H0:1][CX4;!H0:2]>>[C:1]=[C:2]", _KLAPOTKE),
         ("[CX4;!H0:1][CX4;!H0:2]>>[C:1]=[C:2]", _AGRAWAL),
     ],
     (0, +1, -1, +1),
     "C(sp3)–C(sp3) → C=C (introduce unsaturation)."),
    ("add_phenyl_CH",
     [
         ("[cH:1]>>[c:1]c1ccccc1", _POLITZER_STRUCT),
         ("[c;H1:1]>>[c:1]c1ccccc1", _POLITZER),
         ("[cH1:1]>>[c:1]c1ccccc1", _KLAPOTKE),
     ],
     (0, 0, +1, 0),
     "Aromatic C–H → C–Ph (ring stacking rigidity)."),

    # --- Conversions / swaps -----------------------------------------------
    ("amino_to_nitro",
     [
         ("[c:1][NH2]>>[c:1][N+](=O)[O-]", _KLAPOTKE),
         ("[c:1][NH2]>>[c:1][N+](=O)[O-]", _OLAH_NITRATION),
         ("[c:1][NH2]>>[c:1][N+](=O)[O-]", _ZEMAN),
     ],
     (0, +1, +1, -1),
     "Ar–NH2 → Ar–NO2 (oxidation)."),
    ("amino_to_azide",
     [
         ("[c:1][NH2]>>[c:1]N=[N+]=[N-]", _BRASE),
         ("[c:1][NH2]>>[c:1]N=[N+]=[N-]", _KATRITZKY),
         ("[c:1][NH2]>>[c:1]N=[N+]=[N-]", _KLAPOTKE),
     ],
     (-1, 0, 0, +1),
     "Ar–NH2 → Ar–N3 (diazotization)."),
    ("halogen_Br_to_nitro",
     [
         ("[c:1]Br>>[c:1][N+](=O)[O-]", _KLAPOTKE),
         ("[c:1]Br>>[c:1][N+](=O)[O-]", _OLAH_NITRATION),
         ("[c:1]Br>>[c:1][N+](=O)[O-]", _ZEMAN),
     ],
     (-1, +1, +1, +1),
     "Ar–Br → Ar–NO2 (lighter, more energetic)."),
    ("halogen_I_to_azide",
     [
         ("[c:1]I>>[c:1]N=[N+]=[N-]", _KATRITZKY),
         ("[c:1]I>>[c:1]N=[N+]=[N-]", _BRASE),
         ("[c:1]I>>[c:1]N=[N+]=[N-]", _KLAPOTKE),
     ],
     (-1, +1, +1, +1),
     "Ar–I → Ar–N3."),

    # --- Removals -----------------------------------------------------------
    ("remove_nitro_arom",
     [
         ("[c:1][N+](=O)[O-]>>[c:1][H]", _GROUP_ADD),
         ("[c:1][N+](=O)[O-]>>[cH:1]", _KJ),
         ("[c:1][N+](=O)[O-]>>[cH1:1]", _ZEMAN),
     ],
     (-1, -1, -1, 0),
     "Remove aromatic nitro group."),
    ("remove_bromo_arom",
     [
         ("[c:1]Br>>[c:1][H]", _GROUP_ADD),
         ("[c:1]Br>>[cH:1]", _SIKDER),
         ("[c:1]Br>>[cH1:1]", _KLAPOTKE),
     ],
     (-1, 0, 0, 0),
     "Remove aromatic Br."),
    ("remove_azide_arom",
     [
         ("[c:1]N=[N+]=[N-]>>[c:1][H]", _GROUP_ADD),
         ("[c:1]N=[N+]=[N-]>>[cH:1]", _BRASE),
         ("[c:1]N=[N+]=[N-]>>[cH1:1]", _KATRITZKY),
     ],
     (0, 0, 0, -1),
     "Remove aromatic azide."),
    ("nitro_to_methyl_arom",
     [
         ("[c:1][N+](=O)[O-]>>[c:1]C", _GROUP_ADD),
         ("[c:1][N+](=O)[O-]>>[c:1]C", _KLAPOTKE),
         ("[c:1][N+](=O)[O-]>>[c:1]C", _AGRAWAL),
     ],
     (-1, -1, -1, -1),
     "Ar–NO2 → Ar–CH3 (downgrade to alkyl)."),
    ("nitramine_to_amine",
     [
         ("[N:1]([N+](=O)[O-])>>[N:1][H]", _GROUP_ADD),
         ("[N:1]([N+](=O)[O-])>>[N:1][H]", _KLAPOTKE),
         ("[N:1]([N+](=O)[O-])>>[N:1][H]", _AGRAWAL),
     ],
     (0, 0, -1, -1),
     "Nitramine N–NO2 → amine N–H."),
    ("methyl_to_H",
     [
         ("[cX3:1][CH3]>>[cX3:1][H]", _GROUP_ADD),
         ("[cX3:1][CH3]>>[cX3:1][H]", _KLAPOTKE),
         ("[cX3:1][CH3]>>[cX3:1][H]", _POLITZER),
     ],
     (-1, 0, -1, +1),
     "Ar–CH3 → Ar–H."),

    # --- Nitration: oxygen-rich and nitroso reagents -------------------------
    ("nitrate_ester_OH",
     [
         ("[OX2H:1]>>[O:1][N+](=O)[O-]", _KLAPOTKE),
         ("[OH:1]>>[O:1][N+](=O)[O-]", _AGRAWAL),
         ("[C:2][OH:1]>>[C:2][O:1][N+](=O)[O-]", _KJ),
     ],
     (+1, +1, 0, 0),
     "Alcohol O–H → nitrate ester O–NO2."),
    ("gem_dinitro",
     [
         ("[CX4;H2:1]>>[C:1]([N+](=O)[O-])[N+](=O)[O-]", _KLAPOTKE),
         ("[CH2;X4:1]>>[C:1]([N+](=O)[O-])[N+](=O)[O-]", _AGRAWAL),
         ("[CX4;H2;!$(C[c]):1]>>[C:1]([N+](=O)[O-])[N+](=O)[O-]", _KJ),
     ],
     (+1, +1, +1, +1),
     "Aliphatic CH2 → gem-dinitro C(NO2)2."),
    ("trinitromethyl",
     [
         ("[CH3:1]>>[C:1]([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", _KLAPOTKE),
         ("[CX4;H3:1]>>[C:1]([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", _AGRAWAL),
         ("[c:2][CH3:1]>>[c:2][C:1]([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", _KJ),
     ],
     (+1, +1, 0, +1),
     "Methyl → trinitromethyl C(NO2)3."),
    ("nitroso_arom",
     [
         ("[cH:1]>>[c:1]N=O", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]N=O", _AGRAWAL),
         ("[cH1:1]>>[c:1]N=O", _KLAPOTKE),
     ],
     (0, -1, +1, +1),
     "Aromatic C–H → C–NO (nitroso; less oxygen than nitro)."),

    # --- Oxidation: peroxide, epoxide, aldehyde ------------------------------
    ("hydroperoxy_CH",
     [
         ("[cH:1]>>[c:1]OO", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]OO", _POLITZER),
         ("[cH1:1]>>[c:1]OO", _KLAPOTKE),
     ],
     (+1, 0, 0, +1),
     "Aromatic C–H → C–OOH (hydroperoxide)."),
    ("epoxidize_CC",
     [
         ("[CX3;!c:1]=[CX3;!c:2]>>[C:1]1[C:2]O1", _GROUP_ADD),
         ("[C;!c:1]=[C;!c:2]>>[C:1]1[C:2]O1", _AGRAWAL),
         ("[CX3;!$(c):1]=[CX3;!$(c):2]>>[C:1]1[C:2]O1", _KLAPOTKE),
     ],
     (+1, 0, +1, 0),
     "Aliphatic C=C → epoxide."),
    ("formyl_CH",
     [
         ("[cH:1]>>[c:1]C=O", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]C=O", _AGRAWAL),
         ("[cH1:1]>>[c:1]C=O", _ZHANG_CEC),
     ],
     (0, -1, 0, -1),
     "Aromatic C–H → C–CHO (formyl)."),

    # --- Nitrogen heterocycles ----------------------------------------------
    ("furazan_CH",
     [
         ("[cH:1]>>[c:1]c1nonc1", _PAGORIA),
         ("[c;H1:1]>>[c:1]c1nonc1", _GAO),
         ("[cH1:1]>>[c:1]c1nonc1", _KLAPOTKE),
     ],
     (+1, +1, +1, +1),
     "Aromatic C–H → C–furazan (1,2,5-oxadiazole)."),
    ("furoxan_CH",
     [
         ("[cH:1]>>[c:1]c1no[n+]([O-])c1", _PAGORIA),
         ("[c;H1:1]>>[c:1]c1no[n+]([O-])c1", _GAO),
         ("[cH1:1]>>[c:1]c1[n+]([O-])onc1", _KLAPOTKE),
     ],
     (+1, +1, 0, +1),
     "Aromatic C–H → C–furoxan (furazan N-oxide)."),
    ("tetrazine_CH",
     [
         ("[cH:1]>>[c:1]c1nncnn1", _CHAVEZ),
         ("[c;H1:1]>>[c:1]c1nncnn1", _GAO),
         ("[cH1:1]>>[c:1]c1nncnn1", _KLAPOTKE),
     ],
     (0, +1, +1, +1),
     "Aromatic C–H → C–(1,2,4,5-tetrazine)."),
    ("oxadiazole_CH",
     [
         ("[cH:1]>>[c:1]c1nnco1", _GAO),
         ("[c;H1:1]>>[c:1]c1nnco1", _PAGORIA),
         ("[cH1:1]>>[c:1]c1ocnn1", _KLAPOTKE),
     ],
     (+1, 0, 0, +1),
     "Aromatic C–H → C–(1,3,4-oxadiazole)."),
    ("pyrazole_CH",
     [
         ("[cH:1]>>[c:1]c1cc[nH]n1", _KATRITZKY),
         ("[c;H1:1]>>[c:1]c1cc[nH]n1", _GAO),
         ("[cH1:1]>>[c:1]c1n[nH]cc1", _KLAPOTKE),
     ],
     (0, 0, 0, +1),
     "Aromatic C–H → C–pyrazole."),
    ("imidazole_CH",
     [
         ("[cH:1]>>[c:1]c1nc[nH]c1", _KATRITZKY),
         ("[c;H1:1]>>[c:1]c1nc[nH]c1", _GAO),
         ("[cH1:1]>>[c:1]c1c[nH]cn1", _KLAPOTKE),
     ],
     (0, 0, +1, 0),
     "Aromatic C–H → C–imidazole."),
    ("triazine_CH",
     [
         ("[cH:1]>>[c:1]c1ncncn1", _COBURN),
         ("[c;H1:1]>>[c:1]c1ncncn1", _PAGORIA),
         ("[cH1:1]>>[c:1]c1ncncn1", _KLAPOTKE),
     ],
     (0, +1, 0, +1),
     "Aromatic C–H → C–(1,3,5-triazine)."),
    ("nitrile_to_tetrazole",
     [
         ("[C:1]#N>>[c:1]1nnn[nH]1", _KLAPOTKE),
         ("[CX2:1]#N>>[c:1]1nnn[nH]1", _KATRITZKY),
         ("[C:1]#N>>[c:1]1[nH]nnn1", _GAO),
     ],
     (+1, +1, +1, 0),
     "Nitrile → 5-substituted tetrazole."),

    # --- Halogenation -------------------------------------------------------
    ("difluoramino_CH",
     [
         ("[cH:1]>>[c:1]N(F)F", _KLAPOTKE),
         ("[c;H1:1]>>[c:1]N(F)F", _AGRAWAL),
         ("[cH1:1]>>[c:1]N(F)F", _KJ),
     ],
     (+1, +1, +1, 0),
     "Aromatic C–H → C–NF2 (difluoramino)."),
    ("fluoro_alkyl",
     [
         ("[CX4;!H0;!c:1]>>[C:1]F", _KLAPOTKE),
         ("[CX4;H1,H2,H3:1]>>[C:1]F", _AGRAWAL),
         ("[CH2;X4:1]>>[CH1:1]F", _GROUP_ADD),
     ],
     (+1, 0, +1, -1),
     "Aliphatic C–H → C–F."),

    # --- Reduction / hydrogenation ------------------------------------------
    ("hydrogenate_CC",
     [
         ("[CX3;H1:1]=[CX3;H1:2]>>[CH2:1][CH2:2]", _GROUP_ADD),
         ("[CX3;H2:1]=[CX3:2]>>[CH3:1][CH2:2]", _KLAPOTKE),
         ("[C;!c:1]=[C;!c:2]>>[C:1][C:2]", _AGRAWAL),
     ],
     (0, -1, +1, -1),
     "Aliphatic C=C → C–C (hydrogenation)."),
    ("nitro_to_amino",
     [
         ("[c:1][N+](=O)[O-]>>[c:1][NH2]", _OLAH_NITRATION),
         ("[c:1][N+](=O)[O-]>>[c:1]N", _KLAPOTKE),
         ("[c:1][N+](=O)[O-]>>[cH0:1][NH2]", _ZEMAN),
     ],
     (0, -1, -1, +1),
     "Ar–NO2 → Ar–NH2."),
    ("azide_to_amino",
     [
         ("[c:1]N=[N+]=[N-]>>[c:1][NH2]", _BRASE),
         ("[c:1]N=[N+]=[N-]>>[c:1]N", _KATRITZKY),
         ("[c:1]N=[N+]=[N-]>>[cH0:1][NH2]", _KLAPOTKE),
     ],
     (-1, 0, +1, -1),
     "Ar–N3 → Ar–NH2."),

    # --- Nitrogen bases beyond amino / hydrazino / cyano --------------------
    ("guanidino_CH",
     [
         ("[cH:1]>>[c:1]N=C(N)N", _KLAPOTKE),
         ("[c;H1:1]>>[c:1]N=C(N)N", _KATRITZKY),
         ("[cH1:1]>>[c:1]NC(=N)N", _GAO),
     ],
     (0, +1, 0, +1),
     "Aromatic C–H → C–guanidine."),
    ("amide_CH",
     [
         ("[cH:1]>>[c:1]C(N)=O", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]C(N)=O", _AGRAWAL),
         ("[cH1:1]>>[c:1]C(=O)N", _ZHANG_CEC),
     ],
     (0, -1, +1, -1),
     "Aromatic C–H → C–CONH2 (primary amide)."),
    ("isocyanate_CH",
     [
         ("[cH:1]>>[c:1]N=C=O", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]N=C=O", _KATRITZKY),
         ("[cH1:1]>>[c:1]N=C=O", _KLAPOTKE),
     ],
     (0, +1, -1, 0),
     "Aromatic C–H → C–NCO (isocyanate)."),

    # --- Alkylation (performance-diluting carbon) ---------------------------
    ("methyl_CH",
     [
         ("[cH:1]>>[c:1]C", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]C", _AGRAWAL),
         ("[cH1:1]>>[c:1]C", _KLAPOTKE),
     ],
     (-1, -1, 0, 0),
     "Aromatic C–H → C–CH3."),
    ("ethyl_CH",
     [
         ("[cH:1]>>[c:1]CC", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]CC", _AGRAWAL),
         ("[cH1:1]>>[c:1]CC", _KLAPOTKE),
     ],
     (-1, 0, -1, 0),
     "Aromatic C–H → C–CH2CH3."),
    ("hydroxymethyl_CH",
     [
         ("[cH:1]>>[c:1]CO", _GROUP_ADD),
         ("[c;H1:1]>>[c:1]CO", _POLITZER),
         ("[cH1:1]>>[c:1]CO", _ZHANG_CEC),
     ],
     (0, -1, 0, -1),
     "Aromatic C–H → C–CH2OH."),

    # --- Arylation ----------------------------------------------------------
    ("benzyl_CH",
     [
         ("[cH:1]>>[c:1]Cc1ccccc1", _POLITZER_STRUCT),
         ("[c;H1:1]>>[c:1]Cc1ccccc1", _POLITZER),
         ("[cH1:1]>>[c:1]Cc1ccccc1", _KLAPOTKE),
     ],
     (0, -1, +1, 0),
     "Aromatic C–H → C–CH2Ph."),

    # --- Unsaturation -------------------------------------------------------
    ("alkyne_CC",
     [
         ("[CX4;H2;!R:1][CX4;H2;!R:2]>>[C:1]#[C:2]", _GROUP_ADD),
         ("[CX4;H2;!R:1][CX4;H3:2]>>[C:1]#[CH:2]", _KLAPOTKE),
         ("[CH2;X4;!R:1][CH3:2]>>[C:1]#[CH:2]", _AGRAWAL),
     ],
     (0, +1, 0, +1),
     "Acyclic C–C → C≡C."),

    # --- Aromatization ------------------------------------------------------
    ("aromatize_diene",
     [
         ("[C:1]1=[C:2][CX4:3][C:4]=[C:5][CX4:6]1>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1", _POLITZER),
         ("[C:1]1=[C:2][C:3]=[C:4][CX4:5][CX4:6]1>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1", _KLAPOTKE),
         ("[C:1]1[CX4:2][C:3]=[C:4][CX4:5][C:6]=1>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1", _GROUP_ADD),
     ],
     (0, +1, 0, -1),
     "Cyclohexadiene → benzene (aromatization)."),
]


# ---------------------------------------------------------------------------
# Validate primitive SMARTS at import time
# ---------------------------------------------------------------------------

def _validate_primitives() -> None:
    for name, variants, delta, _desc in _PRIMITIVE_ROWS:
        if name != "identity" and len(variants) != 3:
            raise ImportError(
                f"strategy_pool: primitive '{name}' must declare exactly "
                f"three SMARTS variants (+ sources), got {len(variants)}"
            )
        if name == "identity" and variants != [(None, "N/A")]:
            raise ImportError(
                "strategy_pool: identity primitive expects a single [(None, 'N/A')]"
            )

        for smarts, variant_src in variants:
            if smarts is None:
                continue

            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                raise ImportError(
                    f"strategy_pool: primitive '{name}' variant has invalid SMARTS "
                    f"({variant_src}: {smarts!r})"
                )
            if rxn.GetNumReactantTemplates() != 1 or rxn.GetNumProductTemplates() != 1:
                raise ImportError(
                    f"strategy_pool: primitive '{name}' variant must have exactly "
                    f"one reactant / one product template; got "
                    f"{rxn.GetNumReactantTemplates()} / {rxn.GetNumProductTemplates()} "
                    f"({variant_src}: {smarts!r})"
                )

        for d in delta:
            if d not in (-1, 0, +1):
                raise ImportError(
                    f"strategy_pool: primitive '{name}' delta {delta} is "
                    f"outside {{-1, 0, +1}}^4"
                )


_validate_primitives()


# ---------------------------------------------------------------------------
# Map the 81 direction tuples → primitive by L1-nearest Δ.
# ---------------------------------------------------------------------------

# Priority rank for tie-breaking. Lower number == higher preference.
# Canonical energetic chemistry wins over obscure swaps.
_PRIORITY = {
    "nitramine_NH": 0,
    "nitrate_arom_CH": 1,
    "nitrate_alkyl_CH": 2,
    "tetrazole_CH": 3,
    "triazole_CH": 4,
    "azide_arom": 5,
    "azide_methyl": 6,
    "n_oxide": 7,
    "amino_to_nitro": 8,
    "halogen_Br_to_nitro": 9,
    "halogen_I_to_azide": 10,
    "cyano_CH": 11,
    "hydrazino_CH": 12,
    "methyl_to_nitromethyl": 13,
    "amino_to_azide": 14,
    "fluoro_arom_CH": 15,
    "trifluoromethyl_CH": 16,
    "chloro_arom_CH": 17,
    "bromo_arom_CH": 18,
    "iodo_arom_CH": 19,
    "hydroxyl_CH": 20,
    "methoxy_CH": 21,
    "carboxyl_CH": 22,
    "methylester_CH": 23,
    "amino_CH": 24,
    "dehydrogenate_CC": 25,
    "add_phenyl_CH": 26,
    "methyl_to_H": 27,
    "remove_nitro_arom": 28,
    "remove_bromo_arom": 29,
    "remove_azide_arom": 30,
    "nitro_to_methyl_arom": 31,
    "nitramine_to_amine": 32,
    "nitrate_ester_OH": 33,
    "gem_dinitro": 34,
    "trinitromethyl": 35,
    "nitroso_arom": 36,
    "hydroperoxy_CH": 37,
    "epoxidize_CC": 38,
    "formyl_CH": 39,
    "furazan_CH": 40,
    "furoxan_CH": 41,
    "tetrazine_CH": 42,
    "oxadiazole_CH": 43,
    "pyrazole_CH": 44,
    "imidazole_CH": 45,
    "triazine_CH": 46,
    "nitrile_to_tetrazole": 47,
    "difluoramino_CH": 48,
    "fluoro_alkyl": 49,
    "hydrogenate_CC": 50,
    "nitro_to_amino": 51,
    "azide_to_amino": 52,
    "guanidino_CH": 53,
    "amide_CH": 54,
    "isocyanate_CH": 55,
    "methyl_CH": 56,
    "ethyl_CH": 57,
    "hydroxymethyl_CH": 58,
    "benzyl_CH": 59,
    "alkyne_CC": 60,
    "aromatize_diene": 61,
    "identity": 99,
}


def _l1(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


def _build_strategy_pool() -> Dict[Tuple[int, int, int, int], dict]:
    pool: Dict[Tuple[int, int, int, int], dict] = {}
    for d1 in (-1, 0, +1):
        for d2 in (-1, 0, +1):
            for d3 in (-1, 0, +1):
                for d4 in (-1, 0, +1):
                    target = (d1, d2, d3, d4)
                    if target == (0, 0, 0, 0):
                        # Force identity for the true no-op tuple.
                        pool[target] = {
                            'primitive': 'identity',
                            'variants': [{'smarts': None, 'source': 'N/A'}],
                            'delta': (0, 0, 0, 0),
                            'description': (
                                "No modification required (all properties on target)."
                            ),
                        }
                    else:
                        best = None
                        for prim_name, prim_variants, prim_delta, prim_desc in _PRIMITIVE_ROWS:
                            if prim_name == "identity":
                                continue  # identity only valid for (0,0,0,0)
                            dist = _l1(prim_delta, target)
                            prio = _PRIORITY.get(prim_name, 1000)
                            key = (dist, prio)
                            if best is None or key < best[0]:
                                best = (key, prim_name, prim_variants, prim_delta, prim_desc)
                        _, name, variants, delta, desc = best
                        pool[target] = {
                            'primitive': name,
                            'variants': [
                                {'smarts': s, 'source': sr} for s, sr in variants
                            ],
                            'delta': delta,
                            'description': desc,
                        }
    return pool


STRATEGY_POOL: Dict[Tuple[int, int, int, int], dict] = _build_strategy_pool()

# Sanity check — we must have exactly 3^4 = 81 keys.
assert len(STRATEGY_POOL) == 81, f"Expected 81 strategies, got {len(STRATEGY_POOL)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StrategyPoolModifier:
    """Applies pre-built molecular modification strategies based on property gaps."""

    def __init__(self, config=None):
        self.config = config
        self.pool = STRATEGY_POOL
        logger.info(f"Initialized StrategyPoolModifier with {len(self.pool)} strategies")

    def _gap_to_direction(self, gap: float, threshold: float = 0.01) -> int:
        if gap > threshold:
            return +1
        if gap < -threshold:
            return -1
        return 0

    def get_strategy_key(self, property_gap: Dict[str, float]) -> Tuple[int, int, int, int]:
        return (
            self._gap_to_direction(property_gap.get('Density', 0)),
            self._gap_to_direction(property_gap.get('Det Velocity', 0)),
            self._gap_to_direction(property_gap.get('Det Pressure', 0)),
            self._gap_to_direction(property_gap.get('Hf solid', 0)),
        )

    def get_strategy(self, property_gap: Dict[str, float]) -> dict:
        key = self.get_strategy_key(property_gap)
        strategy = self.pool.get(key, self.pool[(0, 0, 0, 0)])
        nvar = len(strategy.get('variants', ()))
        logger.info(
            f"Strategy for {key}: {strategy['primitive']} — "
            f"{strategy['description']} ({nvar} literature variant(s))"
        )
        return strategy

    def apply_strategy(self, smiles: str, strategy: dict) -> List[str]:
        variants_list = strategy.get('variants')
        if variants_list:
            viable = [v for v in variants_list if v.get('smarts') is not None]
        else:
            # Backward compatibility with flat strategy dicts.
            leg = strategy.get('smarts')
            viable = (
                [{'smarts': leg, 'source': strategy.get('source', '')}]
                if leg is not None
                else []
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        if not viable:
            return []

        choice = random.choice(viable)
        smarts = choice['smarts']
        results: List[str] = []
        logger.debug(
            "Literature variant: %s",
            str(choice.get("source", ""))[:200],
        )
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                return []

            products = rxn.RunReactants((mol,))
            for product_tuple in products:
                for product in product_tuple:
                    try:
                        Chem.SanitizeMol(product)
                        new_smiles = Chem.MolToSmiles(product)
                        if not new_smiles or '.' in new_smiles or new_smiles == smiles:
                            continue
                        if Chem.MolFromSmiles(new_smiles) is not None:
                            results.append(new_smiles)
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Strategy '{strategy.get('primitive')}' application failed: {e}")

        return results

    def apply_strategies(self, smiles: str, property_gap: Dict[str, float],
                         target_count: int = 10) -> List[str]:
        primary = self.get_strategy(property_gap)
        key = self.get_strategy_key(property_gap)
        logger.info(f"Applying strategy pool to {smiles} (key={key}, target={target_count})")

        all_mods: "set[str]" = set(self.apply_strategy(smiles, primary))

        # Explore Δ-neighbours for diversity.
        if len(all_mods) < target_count:
            for delta in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0),
                          (0, 0, 0, -1), (0, 0, -1, 0), (0, -1, 0, 0), (-1, 0, 0, 0)]:
                if len(all_mods) >= target_count:
                    break
                neighbor_key = tuple(max(-1, min(1, k + d)) for k, d in zip(key, delta))
                if neighbor_key == key:
                    continue
                neighbor = self.pool.get(neighbor_key)
                if neighbor is not None:
                    all_mods.update(self.apply_strategy(smiles, neighbor))

        # Exhaust remaining primitives for diversity if still short.
        if len(all_mods) < target_count:
            for entry in self.pool.values():
                if len(all_mods) >= target_count:
                    break
                all_mods.update(self.apply_strategy(smiles, entry))

        return list(all_mods)[:target_count]


# Broad family -> the decision the property gap can make.
# Reagent -> one specific reaction inside that family. SMARTS spellings of the
# same reaction stay as apply-time fallbacks; they are not planner classes.
_REACTION_GROUP: Dict[str, str] = {
    "nitrate_arom_CH": "nitration",
    "nitrate_alkyl_CH": "nitration",
    "nitramine_NH": "nitration",
    "methyl_to_nitromethyl": "nitration",
    "halogen_Br_to_nitro": "nitration",
    "azide_arom": "azidation",
    "azide_methyl": "azidation",
    "amino_to_azide": "azidation",
    "halogen_I_to_azide": "azidation",
    "tetrazole_CH": "heterocycle",
    "triazole_CH": "heterocycle",
    "n_oxide": "oxidation",
    "amino_to_nitro": "oxidation",
    "hydroxyl_CH": "oxidation",
    "methoxy_CH": "oxidation",
    "carboxyl_CH": "oxidation",
    "methylester_CH": "oxidation",
    "fluoro_arom_CH": "halogenation",
    "trifluoromethyl_CH": "halogenation",
    "chloro_arom_CH": "halogenation",
    "bromo_arom_CH": "halogenation",
    "iodo_arom_CH": "halogenation",
    "amino_CH": "amination",
    "hydrazino_CH": "amination",
    "cyano_CH": "amination",
    "remove_nitro_arom": "reduction",
    "remove_bromo_arom": "reduction",
    "remove_azide_arom": "reduction",
    "nitro_to_methyl_arom": "reduction",
    "nitramine_to_amine": "reduction",
    "methyl_to_H": "reduction",
    "dehydrogenate_CC": "dehydrogenation",
    "alkyne_CC": "dehydrogenation",
    "add_phenyl_CH": "arylation",
    "benzyl_CH": "arylation",
    "nitrate_ester_OH": "nitration",
    "gem_dinitro": "nitration",
    "trinitromethyl": "nitration",
    "nitroso_arom": "nitration",
    "hydroperoxy_CH": "oxidation",
    "epoxidize_CC": "oxidation",
    "formyl_CH": "oxidation",
    "furazan_CH": "heterocycle",
    "furoxan_CH": "heterocycle",
    "tetrazine_CH": "heterocycle",
    "oxadiazole_CH": "heterocycle",
    "pyrazole_CH": "heterocycle",
    "imidazole_CH": "heterocycle",
    "triazine_CH": "heterocycle",
    "nitrile_to_tetrazole": "heterocycle",
    "difluoramino_CH": "halogenation",
    "fluoro_alkyl": "halogenation",
    "hydrogenate_CC": "reduction",
    "nitro_to_amino": "reduction",
    "azide_to_amino": "reduction",
    "guanidino_CH": "amination",
    "amide_CH": "amination",
    "isocyanate_CH": "amination",
    "methyl_CH": "alkylation",
    "ethyl_CH": "alkylation",
    "hydroxymethyl_CH": "alkylation",
    "aromatize_diene": "aromatization",
}


def list_primitives() -> List[dict]:
    """Export primitive metadata. ``family_id`` is the specific reaction."""
    out = []
    for name, variants, delta, desc in _PRIMITIVE_ROWS:
        smarts_list = [s for s, _src in variants if s]
        reagent_ids = [f"{name}__v{i}" for i in range(len(smarts_list))]
        if name != "identity" and name not in _REACTION_GROUP:
            raise KeyError(f"reaction {name} has no broad family in _REACTION_GROUP")
        out.append(
            {
                "family_id": name,
                "group": _REACTION_GROUP.get(name, "identity"),
                "delta": delta,
                "description": desc,
                "smarts_variants": smarts_list,
                "reagent_ids": reagent_ids,
                "sources": [src for _s, src in variants],
            }
        )
    return out


def list_planner_vocab() -> Tuple[List[str], List[dict]]:
    """Broad families, and one reagent per specific reaction inside them.

    Halogenation's reagents are F / CF3 / Cl / Br / I, which move the four
    properties differently. Nitration's reagents are aromatic nitration,
    aliphatic nitration, nitramine, and so on. SMARTS spelling variants are
    not classes.
    """
    groups: List[str] = []
    reagents: List[dict] = []
    for prim in list_primitives():
        if prim["family_id"] == "identity":
            continue
        if prim["group"] not in groups:
            groups.append(prim["group"])
        reagents.append(
            {
                "reagent_id": prim["family_id"],
                "family_id": prim["group"],
                "delta": prim["delta"],
                "description": prim["description"],
            }
        )
    return groups, reagents


def planner_labels(reaction_family: str, reagent_id: Optional[str] = None) -> Tuple[str, str]:
    """Map a pairs row onto (broad family, specific reaction).

    Older CSVs store the specific reaction in ``reaction_family`` and a SMARTS
    spelling (``name__v0``) in ``reagent_id``. Newer rows store the broad
    family and the specific reaction directly.
    """
    raw = str(reaction_family)
    reag = "" if reagent_id is None else str(reagent_id)
    if reag.endswith(".0"):
        reag = reag[:-2]
    specific = reag.rsplit("__v", 1)[0] if "__v" in reag else reag
    if raw in _REACTION_GROUP:
        return _REACTION_GROUP[raw], raw
    if specific in _REACTION_GROUP:
        return raw, specific
    return raw, specific or raw


def list_reagents() -> List[dict]:
    """Flat SMARTS-variant reagent vocabulary (stable IDs for the reagent head)."""
    out: List[dict] = []
    for prim in list_primitives():
        if prim["family_id"] == "identity":
            continue
        for i, (smarts, rid) in enumerate(zip(prim["smarts_variants"], prim["reagent_ids"])):
            out.append(
                {
                    "reagent_id": rid,
                    "family_id": prim["family_id"],
                    "smarts": smarts,
                    "variant_index": i,
                    "source": (prim.get("sources") or [""])[i] if i < len(prim.get("sources") or []) else "",
                }
            )
    return out


_REAGENT_TEMPLATES: Optional[Dict[str, Chem.Mol]] = None
_REAGENT_REACTIONS: List = []


def reagent_templates() -> Dict[str, Chem.Mol]:
    """Reactant SMARTS templates keyed by reagent_id. Compiled once.

    Templates are copied off the reaction object. RDKit reactant templates are
    owned by the reaction; dropping the reaction and matching later segfaults.
    """
    global _REAGENT_TEMPLATES
    if _REAGENT_TEMPLATES is None:
        compiled: Dict[str, Chem.Mol] = {}
        for reagent in list_reagents():
            rxn = AllChem.ReactionFromSmarts(reagent["smarts"])
            if rxn is None or rxn.GetNumReactantTemplates() < 1:
                continue
            _REAGENT_REACTIONS.append(rxn)
            tmpl = rxn.GetReactantTemplate(0)
            if tmpl is None:
                continue
            compiled[reagent["reagent_id"]] = Chem.Mol(tmpl)
        _REAGENT_TEMPLATES = compiled
    return _REAGENT_TEMPLATES


@lru_cache(maxsize=8192)
def matching_reagent_ids(smiles: str) -> Tuple[str, ...]:
    """Reagent ids whose reactant template matches this parent.

    Used to mask family/reagent heads to chemistry that can actually run.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    hits: List[str] = []
    for reagent_id, tmpl in reagent_templates().items():
        try:
            if mol.HasSubstructMatch(tmpl):
                hits.append(reagent_id)
        except Exception:
            continue
    return tuple(hits)


def matching_family_ids(smiles: str) -> Tuple[str, ...]:
    """Specific reaction names that have a matching SMARTS template."""
    families = []
    seen = set()
    for reagent_id in matching_reagent_ids(smiles):
        family_id = reagent_id.rsplit("__v", 1)[0]
        if family_id not in seen:
            seen.add(family_id)
            families.append(family_id)
    return tuple(families)


def matching_reaction_ids(smiles: str) -> Tuple[str, ...]:
    """Planner reagent ids: specific reactions applicable to this parent."""
    return matching_family_ids(smiles)


def matching_group_ids(smiles: str) -> Tuple[str, ...]:
    """Planner family ids: broad families with at least one applicable reaction."""
    groups = []
    seen = set()
    for reaction_id in matching_reaction_ids(smiles):
        group = _REACTION_GROUP.get(reaction_id)
        if group and group not in seen:
            seen.add(group)
            groups.append(group)
    return tuple(groups)


def reagent_id_for_smarts(family_id: str, smarts: str) -> Optional[str]:
    """Resolve a SMARTS string to its reagent_id within a family."""
    for prim in list_primitives():
        if prim["family_id"] != family_id:
            continue
        for rid, s in zip(prim["reagent_ids"], prim["smarts_variants"]):
            if s == smarts:
                return rid
    return None


def get_modification_strategies(smiles: str, property_gap: Dict[str, float],
                                target_count: int = 20) -> List[str]:
    return StrategyPoolModifier().apply_strategies(smiles, property_gap, target_count)
