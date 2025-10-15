use std::any::Any;

pub trait Component: Any + std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    fn short_name(&self) -> &'static str {
        self.type_name().rsplit("::").next().unwrap_or("Unknown")
    }
}