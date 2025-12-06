# Domain Glossary (Living Document)

This file defines the ubiquitous language of the domain.

---

## Terrain BC

| Term | Definition |
|------|------------|
| **DEM** | Digital Elevation Model — raster grid of elevation values |
| **DSM** | Digital Surface Model — includes buildings/vegetation |
| **GeoPoint** | A WGS84 coordinate (latitude, longitude) |
| **Elevation** | Height above sea level in meters |
| **TerrainGrid** | Immutable 2D array of elevations with CRS and bounds |
| **BoundingBox** | Geographic rectangle (min_x, min_y, max_x, max_y) |
| **CRS** | Coordinate Reference System (e.g., EPSG:4326) |
| **NoData** | Sentinel value indicating missing/invalid pixel |
| **LoS** | Line of Sight — unobstructed path between two points |
| **Geotransform** | Affine transform mapping pixel coords to geographic coords |

---

## Coverage BC

| Term | Definition |
|------|------------|
| **Pathloss** | Signal attenuation between transmitter and receiver (dB) |
| **CoverageMap** | Geographic distribution of signal strength |
| **SignalStrength** | Power level in dBm |
| **Frequency** | Radio frequency in Hz (displayed as MHz/GHz) |
| **Propagation Model** | Mathematical model for RF signal behavior |

---

## Siting BC

| Term | Definition |
|------|------------|
| **CandidateSite** | Potential location for tower installation |
| **Tower** | Existing or planned transmission structure |
| **Constraint** | Rule that must be satisfied for site validity |
| **Score** | Normalized evaluation metric (0.0 to 1.0) |
| **Zone** | Geographic area with specific regulations |

---

## Error Types

| Term | Definition |
|------|------------|
| **TerrainError** | Base exception for terrain operations |
| **InvalidRasterError** | File is not valid raster format |
| **MissingCRSError** | Raster has no CRS defined |
| **InvalidGeotransformError** | Raster has invalid geotransform |
| **AllNoDataError** | Raster contains 100% NoData pixels (unusable) |
| **InvalidBoundsError** | Raster bounds outside valid WGS84 range after reprojection |

---

<!-- glossary v1.1.0 -->
