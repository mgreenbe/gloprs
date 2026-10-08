use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use glop::time_limit::TimeLimit;

fn main() {
    let mut limit = TimeLimit::new(f64::INFINITY, 1.0);
    println!(
        "initial {} {} {}",
        limit.elapsed_deterministic_time(),
        limit.deterministic_time_left(),
        limit.limit_reached()
    );
    limit.advance_deterministic_time(0.25);
    println!(
        "advanced {} {} {}",
        limit.elapsed_deterministic_time(),
        limit.deterministic_time_left(),
        limit.limit_reached()
    );
    limit.advance_deterministic_time(0.75);
    println!(
        "reached {} {} {}",
        limit.elapsed_deterministic_time(),
        limit.deterministic_time_left(),
        limit.limit_reached()
    );
    limit.change_deterministic_limit(2.0);
    println!(
        "extended {} {} {}",
        limit.deterministic_limit(),
        limit.deterministic_time_left(),
        limit.limit_reached()
    );

    let primary = Arc::new(AtomicBool::new(false));
    let secondary = Arc::new(AtomicBool::new(false));
    limit.register_external_limit(Some(Arc::clone(&primary)));
    limit.register_secondary_external_limit(Some(Arc::clone(&secondary)));
    primary.store(true, Ordering::SeqCst);
    println!("primary {}", limit.limit_reached());
    primary.store(false, Ordering::SeqCst);
    secondary.store(true, Ordering::SeqCst);
    println!("secondary {}", limit.limit_reached());
    secondary.store(false, Ordering::SeqCst);
    println!("external_cleared {}", limit.limit_reached());

    let mut global = TimeLimit::new(f64::INFINITY, 0.4);
    let global_external = Arc::new(AtomicBool::new(false));
    global.register_external_limit(Some(Arc::clone(&global_external)));
    limit.merge_with_global_time_limit(Some(&global));
    println!(
        "merged {} {} {} {}",
        limit.elapsed_deterministic_time(),
        limit.deterministic_limit(),
        limit.deterministic_time_left(),
        limit.limit_reached()
    );
    global_external.store(true, Ordering::SeqCst);
    println!("merged_external {}", limit.limit_reached());

    limit.reset_history();
    println!("infinite_time_left {}", limit.time_left().is_infinite());
    let mut zero_wall = TimeLimit::new(0.0, f64::INFINITY);
    println!(
        "zero_wall {} {} {}",
        zero_wall.limit_reached(),
        zero_wall.time_left(),
        zero_wall.limit_reached()
    );
}
