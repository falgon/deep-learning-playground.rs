use super::utils::snd;
use failure::Error;
use ndarray::Array2;
use ndarray_stats::QuantileExt;
use num::Float;
use std::fmt;

pub mod activate_functions;

#[derive(Default)]
pub struct NeuralNetwork<T> {
    neurons: Array2<T>,
}

impl<T: Float + fmt::Display> fmt::Display for NeuralNetwork<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.neurons)
    }
}

impl<T: Float + 'static> NeuralNetwork<T> {
    /// `new` is the constructor of `NeuralNetwork`.
    /// If the height of a given matrix is not 1, `new` returns an `Err`.
    ///
    /// # Arguments
    ///
    /// * `init_neurons` - The initial matrix \\(\mathbb{R}^{1\times n}\\).
    pub fn new(init_neurons: &Array2<T>) -> Result<Self, Error> {
        match init_neurons.dim() {
            (1, _) => Ok(NeuralNetwork::<T> {
                neurons: init_neurons.clone(),
            }),
            _ => Err(failure::format_err!(
                "The shape of initial neurons matrix must be n * 1"
            )),
        }
    }

    /// Let a current matrix \\(X^{1\times m_X}\\),
    /// given arguments \\(W^{n_W\times m_W}\\) (weight) and \\(B^{1\times m_B}\\) (bias)
    /// where \\(m_X=n_W\\), \\(m_W=m_B\\).
    /// Thus, `next` computes next neurons \\(X W+B\\).
    /// If \\(m_X \not = n_W\\) or \\(m_W \not = m_B\\), it returns `Err`.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight matrix \\(W^{n_W\times m_W}\\) for computing next neuron.
    /// * `bias` - Bias matrix \\(B^{1\times m_B}\\) for computing next neuron.
    /// * `activate_function` - The activate function.
    #[inline]
    pub fn next(
        &mut self,
        weight: &Array2<T>,
        bias: &Array2<T>,
        activate_function: &Box<dyn Fn(Array2<T>) -> Array2<T>>,
    ) -> Result<(), Error> {
        match (self.neurons.dim(), weight.dim(), bias.dim()) {
            ((_, width1), (height, width2), (_, width3))
                if width1 == height && width2 == width3 =>
            {
                Ok(self.neurons = activate_function(self.neurons.dot(weight) + bias))
            }
            _ => Err(failure::format_err!("Invalid argument")),
        }
    }

    /// `dim` returns the shape of the array.
    #[inline]
    pub fn dim(&self) -> (ndarray::Ix, ndarray::Ix) {
        self.neurons.dim()
    }

    /// `argmax` returns the index of maximum value.
    #[inline]
    pub fn argmax(&self) -> usize {
        // never panic because the input will not be empty (checking `new` constructor)
        // and the value always defined order (by Float trait).
        snd(self.neurons.argmax().unwrap())
    }
}
