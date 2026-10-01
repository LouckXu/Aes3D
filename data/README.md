# Aesthetic3D scene annotations

[`aesthetic3d_scene_scores.csv`](aesthetic3d_scene_scores.csv) contains **278 scenes**: 7 from Bilarf and 271 from DL3DV. Its `num_views` column sums to **92,649**. Each row represents a scene; individual view annotations are not included.

The supplied table is included unchanged: all 13 columns, row order, and numeric precision are preserved. The loader reads the supplied target values directly, without recomputing them or applying further score normalization.

| Column | Meaning |
| --- | --- |
| `dataset` | Source identifier: `bilarf` or `dl3dv` (DL3DV). |
| `scene_name` | Scene identifier within its source dataset. |
| `num_views` | Number of source views recorded for the scene. |
| `total_score` | Scene-level ArtiMuse overall aesthetic score. Used by `--target total`. |
| `8attr_mean_score` | Unweighted arithmetic mean of the eight aesthetic attributes. Used by `--target 8-attr`, the default. |
| `composition_design` | Composition and design attribute score. |
| `visual_elements_structure` | Visual elements and structure attribute score. |
| `technical_execution` | Technical execution attribute score. |
| `originality_creativity` | Originality and creativity attribute score. |
| `theme_communication` | Theme communication attribute score. |
| `emotion_viewer_response` | Emotion and viewer response attribute score. |
| `overall_gestalt` | Overall gestalt attribute score. |
| `comprehensive_evaluation` | Comprehensive evaluation attribute score; one of the eight attributes, distinct from `total_score`. |

Scores are on a [0, 1] scale and stored to six decimal places. Because each column is rounded separately, recomputing the mean from the displayed attribute values can differ slightly from `8attr_mean_score`; the maximum difference in this table is 0.000000625.

## Computed statistics

Statistics below use all 278 rows with equal weight per scene and population standard deviation.

| Target | Minimum | Maximum | Mean | Median | Standard deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight-attribute mean | 0.179844 | 0.825000 | 0.394286 | 0.366537 | 0.132990 |
| Overall score | 0.195618 | 0.662105 | 0.359242 | 0.339042 | 0.091946 |

The scene count and view count match the manuscript, but the table does not exactly reproduce its eight-attribute statistics (approximately 0.184-0.825, mean 0.395, median 0.369, standard deviation 0.129). The supplement gives an overall-score median of 0.370, whereas this table gives 0.339042. The historical label version and statistic definitions still need reconciliation before claiming numerical reproduction.

The table alone cannot verify view-level aggregation, within-scene score gaps, view-level text descriptions, or human-study results. Those require the underlying per-view records or study data.
