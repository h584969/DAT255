fn main(){
    let model = tch::CModule::load("model.pth").unwrap();
    println!("{:?}", model);
}