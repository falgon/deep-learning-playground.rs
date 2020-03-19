use deep_learning_playground::neural_network::{self, activate_functions};
use deep_learning_playground::setup::dlfs::chap3;
use deep_learning_playground::setup::mnist::{load_data, test_dataset, MnistImage};
use deep_learning_playground::utils::natural_transform::to_io;
use std::env;
use std::io;
use std::time::{Duration, Instant};
use std::vec::Vec;

fn compute(td: &MnistImage, trained_data: &chap3::Chap3Param) -> io::Result<bool> {
    if let Ok(mut nn) = neural_network::NeuralNetwork::<f64>::new(&td.image) {
        let mut afunc = vec![];
        for _ in 1..trained_data.bias.len() {
            afunc.push(activate_functions::sigmoid());
        }
        afunc.push(activate_functions::softmax());

        for ((w, b), af) in trained_data
            .weight
            .iter()
            .zip(trained_data.bias.iter())
            .zip(afunc.iter())
        {
            if let Err(e) = nn.next(&w.map(|x| *x as f64), &b.map(|x| *x as f64), &af) {
                return to_io(Err(e), io::ErrorKind::Other);
            }
        }
        Ok(nn.argmax() == (td.label as usize))
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "failed to construct neural network",
        ))
    }
}

fn execute() -> io::Result<(f64, Duration)> {
    let data = load_data(test_dataset(), true)?;
    println!("Loading success MNIST dataset (size: {})", data.len());

    let trained_data = chap3::load_trained_params()?;
    println!("Loading success trained params");

    let mut accuracy_cnt: u32 = 0;

    println!("Start computing...");
    let start_time = Instant::now();

    for td in data.iter() {
        match compute(&td, &trained_data) {
            Err(e) => return Err(e),
            Ok(t) => {
                if t {
                    accuracy_cnt += 1
                }
            }
        }
    }

    Ok((
        accuracy_cnt as f64 / data.len() as f64,
        start_time.elapsed(),
    ))
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        match execute() {
            Err(e) => eprintln!("{}", e),
            Ok((s, pt)) => println!(
                "Accuracy: {}%, Process time: {}.{:03} seconds",
                s * 100.,
                pt.as_secs(),
                pt.subsec_nanos() / 1_000_000
            ),
        }
    }
}
