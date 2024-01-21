
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::Array2;
use linfa::prelude::*;
use ndarray::prelude ::*;
use std::fs::File;
use std::io::Write;
fn main() {
    let origin_data :  Array2<f32>  = array!(

    
        [1., 1., 1000.0 , 1., 10.], 
        [1., 0., 10. , 1., 6.],
        [1.,0.,0.,1.,6.],
        [1.,0.,0.,1.,6.],
        [1.,0.,0.,1.,6.],
        [1.,0.,800.,1.,8.],
        [1.,0.,800.,1.,8.],
        [1.,0.,800.,1.,8.],
        [0.,1.,0.,0.,3.],
        [0.,1.,50.,0.,3.],
        [0.,1.,10.,1.,3.],
        [0.,0.,0.,0.,2.],
        [0.,0.,10.,0.,1.]
    

    );
    
    let feature_names: Vec<&str> = vec! ["Tv","Pet hound", "Rust LOC","Pasta"] ; 

    let num_features= origin_data.len_of(Axis(1)) -1;

    let featuers: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 2]>> = origin_data.slice(s![..,0..num_features]).to_owned();

    let labels: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 1]>> = origin_data.column(num_features).to_owned();


    let linfa_data = Dataset::new(featuers, labels) 
        .map_targets(|x| match x.to_owned() as i32 {
            i32::MIN..=4 => "Sad",
            5..=7 => "Ok",
            8..=i32::MAX => "Happy",
        }
        ).with_feature_names(feature_names);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_data)
        .unwrap();

    File::create("dt.tex")
    .unwrap()
    .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
    .unwrap();

}

