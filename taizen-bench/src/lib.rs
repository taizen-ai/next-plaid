// next-plaid's features turn on ndarray's `blas`, so `dot` here calls cblas; this
// links next-plaid and with it the BLAS backend it selects.
use next_plaid as _;

pub mod embset;
pub mod ids;
pub mod parity;
pub mod snapshot;
