use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::NumCast;
use polars::prelude::*;

/// Trait for converting a Polars Series to a nalgebra vector (`DMatrix<N>`).
pub trait Series2Nal {
    fn to_nal_vec<N>(&self) -> PolarsResult<DVector<N>>
    where
        N: Scalar + NumCast;
}

impl Series2Nal for Series {
    fn to_nal_vec<N>(&self) -> PolarsResult<DVector<N>>
    where
        N: Scalar + NumCast,
    {
        // Ensure the Series is either numeric or boolean.
        if !(self.dtype().is_numeric() || *self.dtype() == DataType::Boolean) {
            return Err(PolarsError::ComputeError(
                format!("Unsupported data type for conversion: {}", self.dtype()).into(),
            ));
        }

        // Convert Series to DVector<N>.
        let len = self.len();
        let mut data = Vec::with_capacity(len);

        for i in 0..len {
            let val = match self.dtype() {
                DataType::Boolean => match self.bool().unwrap().get(i) {
                    Some(true) => NumCast::from(1).unwrap(),
                    Some(false) => NumCast::from(0).unwrap(),
                    None => NumCast::from(f64::NAN).unwrap(),
                },
                _ => {
                    let f64_val = self.f64().unwrap().get(i).unwrap_or(f64::NAN);
                    NumCast::from(f64_val).unwrap()
                }
            };
            data.push(val);
        }

        Ok(DVector::from(data))
    }
}

/// Trait for converting a Polars DataFrame to a nalgebra matrix (`DMatrix<N>`).
pub trait Df2Nal {
    fn to_nal_mat<N>(&self) -> PolarsResult<DMatrix<N>>
    where
        N: Scalar + NumCast + Copy;
}

impl Df2Nal for DataFrame {
    fn to_nal_mat<N>(&self) -> PolarsResult<DMatrix<N>>
    where
        N: Scalar + NumCast + Copy,
    {
        let nrows = self.height();
        let ncols = self.width();

        if nrows == 0 || ncols == 0 {
            return Err(PolarsError::NoData("DataFrame is empty".into()));
        }

        let mut data: Vec<N> = Vec::with_capacity(nrows * ncols);

        for col in self.get_columns() {
            if !(col.dtype().is_numeric() || *col.dtype() == DataType::Boolean) {
                return Err(PolarsError::ComputeError(
                    format!("Unsupported data type for conversion: {}", col.dtype()).into(),
                ));
            }

            let col_data: DVector<N> = col.to_nal_vec()?;
            // Directly extend data with owned values from col_data
            for val in col_data.iter() {
                data.push(*val); // Dereference to copy the value
            }
        }

        // Now data is Vec<N>, which matches the expected type
        Ok(DMatrix::from_column_slice(nrows, ncols, &data))
    }
}
