use std::any::Any;

pub trait Component: Any + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}