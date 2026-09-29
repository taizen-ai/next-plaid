use anyhow::{bail, Result};

/// IDs safe to pass to next-plaid `delete`: non-empty, unique, 0 <= id < num_documents.
pub fn validate_delete_ids(ids: &[i64], num_documents: usize) -> Result<Vec<i64>> {
    if ids.is_empty() {
        bail!("no ids to delete");
    }
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    for pair in sorted.windows(2) {
        if pair[0] == pair[1] {
            bail!("duplicate id {}", pair[0]);
        }
    }
    if let Some(bad) = sorted.iter().find(|&&id| id < 0 || id as usize >= num_documents) {
        bail!("id {bad} outside 0..{num_documents}");
    }
    Ok(sorted)
}

/// New id of `old_id` after deleting `sorted_deleted` (next-plaid's compaction rule).
pub fn renumber(old_id: i64, sorted_deleted: &[i64]) -> Option<i64> {
    if sorted_deleted.binary_search(&old_id).is_ok() {
        return None;
    }
    Some(old_id - sorted_deleted.partition_point(|&d| d < old_id) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_sorts_and_rejects_bad_ids() {
        assert_eq!(validate_delete_ids(&[5, 1, 3], 10).unwrap(), vec![1, 3, 5]);
        assert!(validate_delete_ids(&[-1], 10).is_err());
        assert!(validate_delete_ids(&[10], 10).is_err());
        assert!(validate_delete_ids(&[2, 2], 10).is_err());
        assert!(validate_delete_ids(&[], 10).is_err());
    }

    #[test]
    fn renumber_matches_compaction() {
        let deleted = [1, 3];
        assert_eq!(renumber(0, &deleted), Some(0));
        assert_eq!(renumber(1, &deleted), None);
        assert_eq!(renumber(2, &deleted), Some(1));
        assert_eq!(renumber(4, &deleted), Some(2));
    }
}
