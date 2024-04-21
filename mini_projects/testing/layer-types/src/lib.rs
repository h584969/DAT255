use std::{marker::PhantomData, ops::{Add, Div, Index, Mul, Sub}, sync::Arc};

use num::{traits::real::Real, Float};

pub trait IsLayer<I, O>{
    fn forward(&self, input: I) -> O;
}

macro_rules! no_param_math_op {
    ($(
        $name:ident => $op:expr
    ),*) => {
        $(
            pub struct $name<Dt: Float>(PhantomData<Dt>);
            impl <I, Dt> IsLayer<I, I> for $name<Dt>
            where I: AsRef<[Dt]> + AsMut<[Dt]> + Clone,
                Dt: Float
            {
                fn forward(&self, input: I) -> I {
                    let mut out = input.clone();
                    for (i, v) in input.as_ref().iter().enumerate(){
                        out.as_mut()[i] = $op(*v)
                    }
                    out
                }
            }
        )*
    };
}

pub struct Linear<const IN: usize, const OUT: usize, Dt>
where Dt: Float
{
    weights: [[Dt;OUT];IN],
    bias: [Dt;OUT]
}

impl<const IN: usize, const OUT: usize, Dt> IsLayer<[Dt;IN], [Dt;OUT]> for Linear<IN, OUT, Dt>
where Dt: Float
{
    fn forward(&self, input: [Dt;IN]) -> [Dt;OUT] {
        let mut out = [Dt::zero();OUT];

        for col in 0..OUT{
            let mut acc = Dt::zero();
            for row in 0..IN{
                acc = acc + input[row]*self.weights[row][col]
            }
        
            out[col] = acc + self.bias[col];
        }

        out
    }
}

no_param_math_op!(
    ReLU => |v: Dt|v.abs(),
    Sigmoid => |v: Dt| (Dt::one()/(Dt::one() - (-v).exp()))
);