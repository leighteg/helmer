use std::marker::PhantomData;

use generational_arena::Index;

pub struct ResourceHandle<T> {
    pub index: Index,
    _phantom: PhantomData<T>,
}

impl<T> ResourceHandle<T> {
    pub fn from_index(index: Index) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }
}

impl<T> Copy for ResourceHandle<T> {}
impl<T> Clone for ResourceHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
