# Data

The analysis uses the two public compilations of NAIC Schedule P loss triangles distributed by the Casualty Actuarial Society, *Loss Reserving Data Pulled from NAIC Schedule P* (Meyers and Shi; the 1998–2007 compilation was prepared for the CAS by S&P Global). They are not redistributed here. Download them from the CAS:

https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p

Save the files under the names below. The checksums identify the versions used in the paper.

**`cas_earlier/`**: "Earlier data sets", accident years 1988–1997 (era A)

| Save as | CAS file | SHA-256 |
|---|---|---|
| `cas_earlier/comauto.csv` | `comauto_pos.csv` | `568a16c666d82e1871658e31ba0280257b846a313c050c6f7eaaaf2d3c009e0b` |
| `cas_earlier/medmal.csv` | `medmal_pos.csv` | `9a5f8cbda1ba1556a22127837212c40c884ebb93cf91292ab7e94ba48c958d63` |
| `cas_earlier/othliab.csv` | `othliab_pos.csv` | `31e7e63091611e2b3e97ee0f30c1e366873147e49abbea6d70ba3cb71301afdd` |
| `cas_earlier/ppauto.csv` | `ppauto_pos.csv` | `d8d06ffbeaba7e6ce896a70417ef58fbc9840b71caffcbe488fb58fa59536ec9` |
| `cas_earlier/prodliab.csv` | `prodliab_pos.csv` | `73cf0b4ca7801dd43687ed4ed2efa4d346f327db10dc849ef2f8f8d6616ea7f1` |
| `cas_earlier/wkcomp.csv` | `wkcomp_pos.csv` | `4c2e23e3dd0edb2a1f3b3434252fa7f19c7bf62c862575c9b0bb87c3ff8606f2` |

**`cas_current/`**: "Current data sets", December 2025, accident years 1998–2007 (era B)

| Save as | CAS file | SHA-256 |
|---|---|---|
| `cas_current/comauto.csv` | `comauto_pos_98-07.csv` | `5012bd4c9048e300669e2f4fc915850449099e159481b4c3349b574f6f00afe1` |
| `cas_current/medmal.csv` | `medmal_pos_98-07.csv` | `50ea237914797562661b82996765e40b1fd09784a63130463bbd4972f74dbba3` |
| `cas_current/othliab.csv` | `othliab_pos_98-07.csv` | `f514136de4be7b5ac114346709c309cd2464861e9fbb683051849f9fa6eadc50` |
| `cas_current/ppauto.csv` | `ppauto_pos98-07.csv` | `6e838f1e44c67218133ef1c8e28ca2b34b9f21ba4ee94de408c412638f798d96` |
| `cas_current/prodliab.csv` | `prodliab_pos_98-07.csv` | `f1070b5a95658bfeb6719fdf4ffdabc8b7e3f97771cdb4b3f503eb7d6e486f01` |
| `cas_current/wkcomp.csv` | `wkcomp_pos_98-07.csv` | `8d0b02bed0939e932f9078f65266e9a398f580f90e5227cde053dd5b520affef` |

To check the downloads: `sha256sum cas_earlier/*.csv cas_current/*.csv` (macOS: `shasum -a 256`).

`python prepare_data.py` then builds `cas_full_square.csv`, the flat era-A panel read by scripts 01 to 04: one row per company group, line, accident year and development lag. Amounts are in USD thousands.
