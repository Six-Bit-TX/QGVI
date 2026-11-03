# Source terms and attribution

This inventory combines assets from several sources. No blanket MIT, Apache,
or Creative Commons license is asserted for all checkpoint and feature files.
The QGVI code license does not replace source-data or upstream-model terms.

| Asset source | Scenes | Declared terms / evidence |
|---|---|---|
| MatrixCity | Block A, Block Small | The [official data card](https://huggingface.co/datasets/BoDai/MatrixCity) declares CC BY-NC 4.0; retain attribution and noncommercial restrictions. |
| UC-GS | NYC, SF | The [official repository](https://github.com/SainingZhang/UC-GS) links synthetic data via Google Drive. A separate explicit dataset redistribution grant was not found in the audited release. |
| HorizonGS | Citysample, Colosseum, Elvenruin, Park, Road | The [official data repository](https://huggingface.co/datasets/BoDai/HorizonGS) has no dataset card or explicit data license in the audited revision. |

The local checkpoint files alone do not establish training ownership or an
independent grant to redistribute scene-derived assets. Repository visibility
remains private pending clarification of those rights.

## Software notices

HorizonGS and LoGS include the Gaussian-Splatting License from Inria and MPII,
which permits research/evaluation use subject to its noncommercial terms and
notice requirements. Unmodified copies are provided in
`licenses/HorizonGS_LICENSE.md` and `licenses/LoGS_LICENSE.md`. These software
licenses are not asserted to be separate permissive dataset licenses.

HLOC has separate Apache-2.0 terms. SuperGlue/SuperPoint have Magic Leap's
noncommercial research restrictions; their code and pretrained weights are
not redistributed here. Obtain them from
[the official provider](https://github.com/magicleap/SuperGluePretrainedNetwork).
NetVLAD network parameters likewise remain an upstream download.

Reference descriptors and triangulated sparse maps are computed scene assets,
not copies of the pretrained neural-network parameters.
