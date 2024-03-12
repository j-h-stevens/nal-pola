/*!
# nal-pola

**nal-pola** provides basic functionality to convert polars dataframes to
nalgebra DVector's and DMatrices. Written for Rust targeting:
* General-purpose dataframe conversion. Contains basic features.

## Using **nal-pola**
You will need the last stable build of the [rust compiler](https://www.rust-lang.org)
and the official package manager: [cargo](https://github.com/rust-lang/cargo).

Simply add the following to your `Cargo.toml` file:

```ignore
[dependencies]
// TODO: replace the * by the latest version.
nal-stats = "*"
```
Most useful functionalities of **nal-pola** are grouped in the root module `nal_pola::`.

However, the recommended way to use **nal-pola** is to import types and traits
explicitly.
*/

pub mod pola2;
pub use pola2::{Df2Nal, Series2Nal};
