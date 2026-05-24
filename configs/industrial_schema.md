# Industrial Dataset Schema

This document describes the expected schema for the proprietary
industrial dataset used in Table 1(a), Table 2, Table 3, and Figure 4
of the paper. The raw data cannot be released, but the schema is
fully documented here so that any practitioner with a comparable
short-form-video recommendation dataset can reproduce the model
with the same configuration in `lara_industrial.yaml`.

## Input features (per impression)

| Group | Columns | Notes |
|-------|---------|-------|
| User  | user_id (hashed), age_bucket, gender, region, device_type, … | discrete; one-hot or embedded |
| Item  | item_id (hashed), creator_id (hashed), category, duration, … | discrete + continuous |
| Context | hour_of_day, day_of_week, network_type, app_position, … | discrete + continuous |

Total feature dimension after one-hot + concatenation: ~1024.

## User-context vector p_u

A separate, low-dimensional (64-d) per-user feature vector consisting of:

- 32-d user-ID embedding (learned in the backbone).
- 32-d aggregated engagement statistics over the past 7 days
  (click count, dwell sum, conversion count, two engagement scores).

## Labels (5 binary tasks)

| Index | Task name      | Positive rate |
|-------|----------------|---------------|
| 0     | click          | ~42%          |
| 1     | dwell time     | ~37%          |
| 2     | business       | ~3%           |
| 3     | engagement_1   | ~12%          |
| 4     | engagement_2   | ~8%           |

`dwell time` is binarized with a fixed threshold (per-domain). All five
labels are observed on every impression.

## Train/test split

Temporal holdout: 29 days for training, the final day for evaluation
(no overlap of impressions or users).
