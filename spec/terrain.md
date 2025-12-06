# BC: terrain — Domain Exploration

The terrain bounded context represents the physical world:
DEM/DSM, elevation, slope, LoS, CRS handling.

Open Questions:
- What DEM/DSM sources will we support initially?
- What CRS must the system normalize to?
- How will TerrainGrid be represented?
- What invariants must elevation data satisfy?
- What constitutes valid or invalid terrain input?
- How will we compute Line-of-Sight in this domain?
- Do we need slope/aspect now or later?
