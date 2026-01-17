use std::any::Any;

/// A marker trait for any type that can be stored as a resource in the ECS.
/// It requires the type to be `Any` (which implies 'static), `Send`, and `Sync`.
pub trait Resource: Any + Send + Sync {}

/// A blanket implementation that automatically makes any compatible type a valid resource.
impl<T: Any + Send + Sync> Resource for T {}
