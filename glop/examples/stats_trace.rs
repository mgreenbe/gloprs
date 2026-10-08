use glop::stats::{DistributionKind, StatsGroup};

fn main() {
    let mut stats = StatsGroup::new("TraceStats");
    for value in [0.125, 0.5, 0.875] {
        stats.add("ratio", DistributionKind::Ratio, value);
    }
    for value in [-1.25e-9, 3.5e4, 2.0] {
        stats.add("value", DistributionKind::Double, value);
    }
    for value in [-2.0, 7.0, 10.0] {
        stats.add("count", DistributionKind::Integer, value);
    }
    print!("{}", stats.stat_string());
    stats.reset();
    println!("after_reset {}", stats.stat_string().len());
}
