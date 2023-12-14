pub mod data;
mod series_data;
pub mod utils;
pub use series_data::*;
pub use utils::*;

pub mod prelude {
    pub use crate::data::generators::*;
    pub use crate::data::*;
    pub use crate::series_data::*;
    pub use crate::utils::*;
}
