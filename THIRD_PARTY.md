# Third-party notices

The `research/original` snapshot contains the project's own research modules; dependencies are installed separately. Some modules retain historical imports and inactive adapters for provenance. Their presence does not make those transports or older algorithms the default.

| Dependency | Source | Terms / handling |
|---|---|---|
| HorizonGS | https://github.com/InternRobotics/HorizonGS | Gaussian-Splatting research/noncommercial license; retained upstream; frozen checkpoints and source data retain their source terms |
| LoGS | https://github.com/RPL-CS-UCL/gs_localization | Inria/Gaussian-Splatting terms and component-specific licenses; downloaded separately |
| hloc | https://github.com/cvg/Hierarchical-Localization | Apache-2.0; individual models have their own terms |
| SuperPoint / SuperGlue | https://github.com/magicleap/SuperGluePretrainedNetwork | Magic Leap noncommercial terms; do not treat upstream code/weights as MIT. Installed from upstream, not mirrored in this repository |
| NetVLAD | https://github.com/uzh-rpg/netvlad_tf_open | Original model provider and its terms; downloaded by the upstream hloc extractor |
| COLMAP / pycolmap | https://github.com/colmap/colmap | BSD-licensed source and dependency notices |
| MatrixCity | https://github.com/city-super/MatrixCity | CC BY-NC 4.0 dataset terms |
| UC-GS | https://github.com/MatrixCity/UC-GS | Consult the original release for scene-data terms; no blanket redistribution license is invented here |

Public images are research demonstrations from these scene assets, not an independently captured dataset. Gaussian renderings may contain reconstruction artifacts. Five scene scales are estimates from DA3; do not represent them as surveyed metric ground truth.

No general MIT/Apache license is asserted for this combined research release. Obtain the relevant authors' permission for uses outside the applicable source terms. GitHub visibility is not a new license for third-party materials.
