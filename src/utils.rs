use std::hash::Hash;

use fnv::FnvHashSet as HashSet;

pub struct Accumulate<I, F>
where
    I: Iterator,
    F: Fn(I::Item, I::Item) -> I::Item,
{
    accum: Option<I::Item>,
    underlying: I,
    acc_fn: F,
}

impl<I, F> Iterator for Accumulate<I, F>
where
    I: Iterator,
    I::Item: Hash + Eq + Clone,
    F: Fn(I::Item, I::Item) -> I::Item,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.next() {
            Some(x) => {
                let new_accum = match self.accum.clone() {
                    Some(accum) => (self.acc_fn)(accum, x),
                    None => x,
                };
                self.accum = Some(new_accum.clone());
                Some(new_accum)
            }
            None => None,
        }
    }
}

pub trait Accumulator: Iterator {
    fn accumulate<F>(self, f: F) -> Accumulate<Self, F>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item,
        Self::Item: Clone,
        Self: Sized,
        Self: Iterator,
    {
        Accumulate {
            accum: None,
            underlying: self,
            acc_fn: f,
        }
    }
}

impl<I: Iterator> Accumulator for I {}

// Returns true if all items in a sequence are distinct
pub fn all_unique<T>(iter: T) -> bool
where
    T: IntoIterator + Clone,
    T::Item: Clone + Eq + Hash,
{
    let mut uniq = HashSet::default();
    iter.into_iter().all(move |x| uniq.insert(x))
}

pub fn pairwise<I>(sequence: I) -> impl Iterator<Item = (I::Item, I::Item)>
where
    I: IntoIterator + Clone,
{
    let second_iter = sequence.clone().into_iter().skip(1);
    sequence.into_iter().zip(second_iter)
}

pub fn prepend_vec_append<T>(a: T, sequence: Vec<T>, z: T) -> Vec<T> {
    let mut vec_a = vec![a];
    vec_a.extend(sequence);
    vec_a.extend(vec![z]);
    vec_a
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_accumulator_cumulative() {
        let result: Vec<i32> = (1..6).accumulate(|x, y| x + y).collect();
        let expected_result: Vec<i32> = vec![1, 3, 6, 10, 15];
        assert!(result == expected_result)
    }

    #[test]
    fn test_accumulator_rolling_factorial() {
        let result: Vec<i32> = (1..6).accumulate(|x, y| x * y).collect();
        let expected_result: Vec<i32> = vec![1, 2, 6, 24, 120];
        assert!(result == expected_result)
    }
}
