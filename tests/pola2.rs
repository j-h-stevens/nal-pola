use nal_pola::{Df2Nal, Series2Nal};
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use polars::prelude::*;

    #[test]
    fn test_numeric_series_to_dvector() {
        let series = Series::new("numeric", vec![1.0, 2.0, 3.0]);
        let result = series.to_nal_vec::<f64>().unwrap();
        assert_eq!(result, DVector::from_vec(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_boolean_series_to_dvector() {
        let series = Series::new("boolean", vec![true, false, true]);
        let result = series.to_nal_vec::<f64>().unwrap();
        assert_eq!(result, DVector::from_vec(vec![1.0, 0.0, 1.0]));
    }

    #[test]
    fn test_series_with_nulls_to_dvector() {
        let series = Series::new("nulls", &[Some(1.0), None, Some(3.0)]);
        let result = series.to_nal_vec::<f64>().unwrap();
        // Use assert_relative_eq or a similar method to handle NaN comparisons
        for (a, b) in result.iter().zip(vec![1.0, f64::NAN, 3.0].iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_empty_series_to_dvector() {
        let series = Series::new_empty("empty", &DataType::Float64);
        let result = series.to_nal_vec::<f64>().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dataframe_to_dmatrix() {
        let df = df![
            "numeric" => vec![1.0, 2.0, 3.0],
            "boolean" => vec![true, false, true]
        ]
        .unwrap();
        let result = df.to_nal_mat::<f64>().unwrap();
        let expected = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 0.0, 3.0, 1.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_dataframe_to_dmatrix() {
        let df = DataFrame::default();
        let result = df.to_nal_mat::<f64>();
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "Unsupported data type for conversion: str")]
    fn test_non_numeric_series_to_dvector() {
        let series = Series::new("non_numeric", vec!["a", "b", "c"]);
        let _ = series.to_nal_vec::<f64>().unwrap(); // Make sure to call unwrap to trigger the panic.
    }

    #[test]
    #[should_panic(expected = "Unsupported data type for conversion: str")]
    fn test_dataframe_with_non_numeric_column_to_dmatrix() {
        let df = df![
            "numeric" => vec![1.0, 2.0, 3.0],
            "non_numeric" => vec!["a", "b", "c"]
        ]
        .unwrap();
        let _ = df.to_nal_mat::<f64>().unwrap(); // Make sure to call unwrap to trigger the panic.
    }
}
