//! MiningConfiguration bytes derived from proof row/column index patterns.
//! Mirrors zk-pow `PeriodicPattern::from_list` + `MiningConfiguration::to_bytes`.

const NUM_DIMS: usize = 3;
const RESERVED: [u8; 32] = [0u8; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PeriodicPattern {
    shape: [(u32, u32); NUM_DIMS],
}

impl PeriodicPattern {
    fn from_list(pattern: &[u32]) -> Result<Self, String> {
        if pattern.is_empty() {
            return Err("pattern parsing error: empty".into());
        }
        if !pattern.windows(2).all(|w| w[0] < w[1]) {
            return Err("pattern parsing error: not strictly increasing".into());
        }
        if pattern[0] != 0 {
            return Err("pattern parsing error: must start at 0".into());
        }

        let mut p = pattern.to_vec();
        let mut shape_vec = Vec::new();

        while p.len() > 1 {
            let mut found = false;
            for period in 1..p.len() {
                if !p.len().is_multiple_of(period) {
                    continue;
                }
                let s = p[period];
                let is_periodic = (0..p.len() - period).all(|i| p[i] + s == p[i + period]);
                if is_periodic {
                    shape_vec.push((s, (p.len() / period) as u32));
                    p.truncate(period);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err("pattern parsing error: not periodic".into());
            }
        }

        shape_vec.reverse();
        let period = shape_vec.last().map_or(1, |&(s, l)| s * l);
        while shape_vec.len() < NUM_DIMS {
            shape_vec.push((period, 1));
        }

        let shape: [(u32, u32); NUM_DIMS] = shape_vec.try_into().unwrap();
        let pat = Self { shape };
        if !pat.is_valid() {
            return Err("pattern parsing error: invalid constructed pattern".into());
        }
        Ok(pat)
    }

    fn is_valid(&self) -> bool {
        Self::from_bytes(&self.to_bytes())
            .map(|restored| restored.shape == self.shape)
            .unwrap_or(false)
    }

    fn from_bytes(data: &[u8; 6]) -> Result<Self, String> {
        let mut shape = [(0u32, 0u32); NUM_DIMS];
        let mut min_stride = 1u32;
        for i in 0..NUM_DIMS {
            let factor = data[2 * i] as u32 + 1;
            let length = data[2 * i + 1] as u32 + 1;
            let stride = factor * min_stride;
            if stride > (1 << 24) / length {
                return Err("pattern parsing error: period overflow".into());
            }
            shape[i] = (stride, length);
            min_stride = stride * length;
        }
        Ok(Self { shape })
    }

    fn to_bytes(&self) -> [u8; 6] {
        let mut data = [0u8; 6];
        let mut min_stride = 1u32;
        for (i, &(stride, length)) in self.shape.iter().enumerate() {
            let factor = stride / min_stride;
            data[2 * i] = (factor - 1) as u8;
            data[2 * i + 1] = (length - 1) as u8;
            min_stride = stride * length;
        }
        data
    }

    fn offset_is_valid(&self, mut offset: u32) -> bool {
        for &(stride, length) in self.shape.iter().rev() {
            offset %= stride * length;
            if offset >= stride {
                return false;
            }
        }
        true
    }
}

pub fn mining_config_bytes(
    k: u32,
    rank: u16,
    row_offsets: &[u32],
    col_offsets: &[u32],
) -> Result<[u8; 52], String> {
    let rows_pattern = PeriodicPattern::from_list(row_offsets)?;
    let cols_pattern = PeriodicPattern::from_list(col_offsets)?;

    let mut bytes = [0u8; 52];
    bytes[0..4].copy_from_slice(&k.to_le_bytes());
    bytes[4..6].copy_from_slice(&rank.to_le_bytes());
    bytes[6..8].copy_from_slice(&0u16.to_le_bytes()); // MMAType::Int7xInt7ToInt32
    bytes[8..14].copy_from_slice(&rows_pattern.to_bytes());
    bytes[14..20].copy_from_slice(&cols_pattern.to_bytes());
    bytes[20..52].copy_from_slice(&RESERVED);
    Ok(bytes)
}

pub fn validate_tile_anchor(
    row_offsets: &[u32],
    col_offsets: &[u32],
    t_rows: u32,
    t_cols: u32,
) -> Result<(), String> {
    let rows_pattern = PeriodicPattern::from_list(row_offsets)?;
    let cols_pattern = PeriodicPattern::from_list(col_offsets)?;
    if !rows_pattern.offset_is_valid(t_rows) {
        return Err(format!(
            "t_rows {t_rows} invalid for row pattern (cutlass tile anchor)"
        ));
    }
    if !cols_pattern.offset_is_valid(t_cols) {
        return Err(format!(
            "t_cols {t_cols} invalid for col pattern (cutlass tile anchor)"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bzminer_patterns_encode() {
        let rows: Vec<u32> = vec![0, 8, 32, 40, 64, 72, 96, 104];
        let cols: Vec<u32> = vec![
            0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225,
        ];
        let cfg = mining_config_bytes(4096, 128, &rows, &cols).expect("bzminer cfg");
        assert_eq!(cfg.len(), 52);
    }

    #[test]
    fn contiguous_config_bytes_match_embedded() {
        let rows: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let cols: Vec<u32> = (0..16).map(|i| i as u32).collect();
        let cfg = mining_config_bytes(4096, 128, &rows, &cols).expect("contiguous cfg");
        const EMBEDDED: [u8; 52] = [
            0x00, 0x10, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(cfg, EMBEDDED);
    }

    #[test]
    fn contiguous_8x8_config_bytes_match_embedded() {
        let rows: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let cols: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let cfg = mining_config_bytes(4096, 128, &rows, &cols).expect("contiguous 8x8 cfg");
        const EMBEDDED: [u8; 52] = [
            0x00, 0x10, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(cfg, EMBEDDED);
    }

    #[test]
    fn cutlass_config_bytes_match_embedded() {
        let rows: Vec<u32> = vec![0, 1, 2, 3, 16, 17, 18, 19];
        let cols: Vec<u32> = vec![0, 1, 2, 3, 32, 33, 34, 35];
        let cfg = mining_config_bytes(4096, 128, &rows, &cols).expect("cutlass cfg");
        const EMBEDDED: [u8; 52] = [
            0x00, 0x10, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x03, 0x03, 0x01, 0x00, 0x00,
            0x00, 0x03, 0x07, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(cfg, EMBEDDED);
    }
}
