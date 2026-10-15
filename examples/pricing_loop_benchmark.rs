//! Exercise the top-31 heap construction used by GLOP's pricing cache.
use gloprs::pricing::DynamicMaximum;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let size: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let repetitions: u32 = std::env::args().nth(2).unwrap().parse().unwrap();
    assert!(size > 0 && repetitions > 0);
    let mut pricing = DynamicMaximum::new(1);
    pricing.clear_and_resize(size);
    let mut checksum = 0_usize;
    let start = Instant::now();
    for _ in 0..repetitions {
        pricing.start_dense_updates();
        for i in 0..size {
            pricing.dense_add_or_update(i, f64::from(u32::try_from((i * 97) % size).unwrap()));
        }
        checksum = checksum.wrapping_add(black_box(&mut pricing).get_maximum().unwrap());
    }
    println!(
        "{:.17} {checksum}",
        start.elapsed().as_secs_f64() * 1e9 / f64::from(repetitions)
    );
}
