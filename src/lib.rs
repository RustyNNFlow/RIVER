#[macro_use]
extern crate lazy_static;
extern crate serde;
pub mod addons;
pub mod data;
pub mod datasets;
pub mod modules;
pub mod vision;
pub mod core;
mod error;
pub use error::TchError;

pub(crate) mod wrappers;
pub use wrappers::device::{Cuda, Device};
pub use wrappers::jit::{self, CModule, IValue, TrainableCModule};
pub use wrappers::kind::{self, Kind};
pub use wrappers::optimizer::COptimizer;
pub use wrappers::scalar::Scalar;
pub use wrappers::{
    get_num_interop_threads, get_num_threads, manual_seed, set_num_interop_threads,
    set_num_threads, QEngine,
};

mod tensor;
pub use tensor::{
    autocast, index, no_grad, no_grad_guard, with_grad, IndexOp, NewAxis, NoGradGuard, Reduction,
    Shape, Tensor, TensorIndexer,
};

pub mod nn;
pub mod models;

pub fn maybe_init_cuda() {
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
}
