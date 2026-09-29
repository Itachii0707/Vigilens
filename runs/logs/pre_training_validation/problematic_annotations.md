# Dataset Validation & Problematic Annotation Report

- **Overall Status**: PASSED
- **Total Images**: 128
- **Total Annotations**: 929
- **Data Leakage Issues**: 0
- **Corrupt Images**: 0

## Identified Issues

| Category | File | Line | Details |
| --- | --- | --- | --- |
| extremely_large_bbox | `000000000247.txt` | 4 | Normalized area 0.9821 above 0.98 |
| extremely_small_bbox | `000000000257.txt` | 31 | Normalized area 0.000068 below 0.0001 |
| extremely_small_bbox | `000000000257.txt` | 33 | Normalized area 0.000092 below 0.0001 |
| extremely_large_bbox | `000000000450.txt` | 4 | Normalized area 0.9978 above 0.98 |
| extremely_small_bbox | `000000000531.txt` | 6 | Normalized area 0.000017 below 0.0001 |
| extremely_small_bbox | `000000000531.txt` | 14 | Normalized area 0.000060 below 0.0001 |
| extremely_small_bbox | `000000000540.txt` | 7 | Normalized area 0.000013 below 0.0001 |
| extremely_small_bbox | `000000000540.txt` | 18 | Normalized area 0.000092 below 0.0001 |
| extremely_small_bbox | `000000000542.txt` | 14 | Normalized area 0.000040 below 0.0001 |
| extremely_large_bbox | `000000000605.txt` | 1 | Normalized area 0.9951 above 0.98 |
