# ADR 0001: Subject-wise GroupKFold

Sleep staging epochs from the same subject/night are temporally correlated. Random epoch splits leak adjacent epochs into train and test, inflating metrics. We split by `subject_id` using `GroupKFold`, persist folds to disk, and enforce disjoint train/val/test subject sets with an automated test.
