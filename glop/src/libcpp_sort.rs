//! libc++'s comparator-based `std::sort` ordering for tie-sensitive GLOP paths.
//!
//! GLOP's singleton crash sorts columns by a floating penalty without an
//! index tie-breaker. Rust's slice sort uses a different algorithm and hence
//! visits equal-penalty columns in a different order. This small, index-based
//! port preserves the pinned libc++ introsort's traversal and swaps. Algorithm
//! adapted from LLVM libc++ `__algorithm/sort.h` and its heap helpers
//! (Apache-2.0 WITH LLVM-exception).

use std::cmp::Ordering;

fn less<T: Copy>(values: &[T], a: usize, b: usize, compare: &impl Fn(T, T) -> Ordering) -> bool {
    compare(values[a], values[b]) == Ordering::Less
}

fn sort3<T: Copy>(
    values: &mut [T],
    x: usize,
    y: usize,
    z: usize,
    compare: &impl Fn(T, T) -> Ordering,
) {
    if !less(values, y, x, compare) {
        if !less(values, z, y, compare) {
            return;
        }
        values.swap(y, z);
        if less(values, y, x, compare) {
            values.swap(x, y);
        }
        return;
    }
    if less(values, z, y, compare) {
        values.swap(x, z);
        return;
    }
    values.swap(x, y);
    if less(values, z, y, compare) {
        values.swap(y, z);
    }
}

fn sort4<T: Copy>(values: &mut [T], first: usize, compare: &impl Fn(T, T) -> Ordering) {
    sort3(values, first, first + 1, first + 2, compare);
    for position in (first..first + 3).rev() {
        if less(values, position + 1, position, compare) {
            values.swap(position, position + 1);
        } else {
            break;
        }
    }
}

fn sort5<T: Copy>(values: &mut [T], first: usize, compare: &impl Fn(T, T) -> Ordering) {
    sort4(values, first, compare);
    for position in (first..first + 4).rev() {
        if less(values, position + 1, position, compare) {
            values.swap(position, position + 1);
        } else {
            break;
        }
    }
}

fn insertion<T: Copy>(
    values: &mut [T],
    first: usize,
    last: usize,
    compare: &impl Fn(T, T) -> Ordering,
) {
    for i in first + 1..last {
        let value = values[i];
        let mut j = i;
        while j > first && compare(value, values[j - 1]) == Ordering::Less {
            values[j] = values[j - 1];
            j -= 1;
        }
        values[j] = value;
    }
}

fn incomplete_insertion<T: Copy>(
    values: &mut [T],
    first: usize,
    last: usize,
    compare: &impl Fn(T, T) -> Ordering,
) -> bool {
    if last - first <= 5 {
        match last - first {
            0 | 1 => {}
            2 => {
                if less(values, first + 1, first, compare) {
                    values.swap(first, first + 1);
                }
            }
            3 => sort3(values, first, first + 1, first + 2, compare),
            4 => sort4(values, first, compare),
            5 => sort5(values, first, compare),
            _ => unreachable!(),
        }
        return true;
    }
    sort3(values, first, first + 1, first + 2, compare);
    let mut count = 0;
    for i in first + 3..last {
        if less(values, i, i - 1, compare) {
            let value = values[i];
            let mut j = i;
            loop {
                values[j] = values[j - 1];
                j -= 1;
                if j == first || compare(value, values[j - 1]) != Ordering::Less {
                    break;
                }
            }
            values[j] = value;
            count += 1;
            if count == 8 {
                return i + 1 == last;
            }
        }
    }
    true
}

fn partition_right<T: Copy>(
    values: &mut [T],
    first: usize,
    last: usize,
    compare: &impl Fn(T, T) -> Ordering,
) -> (usize, bool) {
    let pivot = values[first];
    let mut left = first + 1;
    while compare(values[left], pivot) == Ordering::Less {
        left += 1;
    }
    let mut right = last;
    if left == first + 1 {
        while left < right {
            right -= 1;
            if compare(values[right], pivot) == Ordering::Less {
                break;
            }
        }
    } else {
        loop {
            right -= 1;
            if compare(values[right], pivot) == Ordering::Less {
                break;
            }
        }
    }
    let already = left >= right;
    while left < right {
        values.swap(left, right);
        left += 1;
        while compare(values[left], pivot) == Ordering::Less {
            left += 1;
        }
        loop {
            right -= 1;
            if compare(values[right], pivot) == Ordering::Less {
                break;
            }
        }
    }
    let position = left - 1;
    if position != first {
        values[first] = values[position];
    }
    values[position] = pivot;
    (position, already)
}

fn partition_left<T: Copy>(
    values: &mut [T],
    first: usize,
    last: usize,
    compare: &impl Fn(T, T) -> Ordering,
) -> usize {
    let pivot = values[first];
    let mut left = first;
    if compare(pivot, values[last - 1]) == Ordering::Less {
        loop {
            left += 1;
            if compare(pivot, values[left]) == Ordering::Less {
                break;
            }
        }
    } else {
        while left + 1 < last && compare(pivot, values[left + 1]) != Ordering::Less {
            left += 1;
        }
        left += 1;
    }
    let mut right = last;
    if left < right {
        loop {
            right -= 1;
            if compare(pivot, values[right]) != Ordering::Less {
                break;
            }
        }
    }
    while left < right {
        values.swap(left, right);
        loop {
            left += 1;
            if compare(pivot, values[left]) == Ordering::Less {
                break;
            }
        }
        loop {
            right -= 1;
            if compare(pivot, values[right]) != Ordering::Less {
                break;
            }
        }
    }
    let position = left - 1;
    if position != first {
        values[first] = values[position];
    }
    values[position] = pivot;
    left
}

fn heap_sift_down<T: Copy>(
    values: &mut [T],
    first: usize,
    len: usize,
    start: usize,
    compare: &impl Fn(T, T) -> Ordering,
) {
    if len < 2 || start > (len - 2) / 2 {
        return;
    }
    let top = values[first + start];
    let mut hole = start;
    let mut child = 2 * hole + 1;
    if child + 1 < len && less(values, first + child, first + child + 1, compare) {
        child += 1;
    }
    if less(values, first + child, first + hole, compare) {
        return;
    }
    loop {
        values[first + hole] = values[first + child];
        hole = child;
        if hole > (len - 2) / 2 {
            break;
        }
        child = 2 * hole + 1;
        if child + 1 < len && less(values, first + child, first + child + 1, compare) {
            child += 1;
        }
        if compare(values[first + child], top) == Ordering::Less {
            break;
        }
    }
    values[first + hole] = top;
}

fn heap_sort<T: Copy>(
    values: &mut [T],
    first: usize,
    last: usize,
    compare: &impl Fn(T, T) -> Ordering,
) {
    let len = last - first;
    if len <= 1 {
        return;
    }
    for parent in (0..=(len - 2) / 2).rev() {
        heap_sift_down(values, first, len, parent, compare);
    }
    for size in (2..=len).rev() {
        let top = values[first];
        let mut hole = 0;
        loop {
            let mut child = 2 * hole + 1;
            if child + 1 < size && less(values, first + child, first + child + 1, compare) {
                child += 1;
            }
            values[first + hole] = values[first + child];
            hole = child;
            if hole > (size - 2) / 2 {
                break;
            }
        }
        if hole == size - 1 {
            values[first + hole] = top;
        } else {
            values[first + hole] = values[first + size - 1];
            values[first + size - 1] = top;
            let value = values[first + hole];
            while hole != 0 {
                let parent = (hole - 1) / 2;
                if compare(values[first + parent], value) != Ordering::Less {
                    break;
                }
                values[first + hole] = values[first + parent];
                hole = parent;
            }
            values[first + hole] = value;
        }
    }
}

fn introsort<T: Copy>(
    values: &mut [T],
    mut first: usize,
    mut last: usize,
    compare: &impl Fn(T, T) -> Ordering,
    mut depth: usize,
    mut leftmost: bool,
) {
    loop {
        let len = last - first;
        if len <= 1 {
            return;
        }
        if len <= 5 {
            match len {
                2 if less(values, first + 1, first, compare) => {
                    values.swap(first, first + 1);
                }
                3 => sort3(values, first, first + 1, first + 2, compare),
                4 => sort4(values, first, compare),
                5 => sort5(values, first, compare),
                _ => {}
            }
            return;
        }
        if len < 24 {
            insertion(values, first, last, compare);
            return;
        }
        if depth == 0 {
            heap_sort(values, first, last, compare);
            return;
        }
        depth -= 1;
        let middle = first + len / 2;
        if len > 128 {
            sort3(values, first, middle, last - 1, compare);
            sort3(values, first + 1, middle - 1, last - 2, compare);
            sort3(values, first + 2, middle + 1, last - 3, compare);
            sort3(values, middle - 1, middle, middle + 1, compare);
            values.swap(first, middle);
        } else {
            sort3(values, middle, first, last - 1, compare);
        }
        if !leftmost && compare(values[first - 1], values[first]) != Ordering::Less {
            first = partition_left(values, first, last, compare);
            continue;
        }
        let (pivot, already) = partition_right(values, first, last, compare);
        if already {
            let first_sorted = incomplete_insertion(values, first, pivot, compare);
            let second_sorted = incomplete_insertion(values, pivot + 1, last, compare);
            if first_sorted && second_sorted {
                return;
            }
            if first_sorted {
                first = pivot + 1;
                continue;
            }
            if second_sorted {
                last = pivot;
                continue;
            }
        }
        introsort(values, first, pivot, compare, depth, leftmost);
        first = pivot + 1;
        leftmost = false;
    }
}

pub(crate) fn sort_by<T: Copy>(values: &mut [T], compare: impl Fn(T, T) -> Ordering) {
    if values.len() <= 1 {
        return;
    }
    let depth = 2 * (usize::BITS - 1 - values.len().leading_zeros()) as usize;
    introsort(values, 0, values.len(), &compare, depth, true);
}

#[cfg(test)]
mod tests {
    use super::sort_by;

    #[test]
    fn matches_pinned_libcpp_equal_key_orders() {
        // Generated by tools/sort_reference_adapter.cc with Apple libc++.
        for (n, mode, expected) in [
            (23, 2, 5_866_226_670_624_721_064_u64),
            (24, 1, 7_860_708_549_496_834_645),
            (30, 3, 7_819_363_854_602_500_414),
            (100, 2, 11_570_356_855_097_696_211),
            (128, 0, 7_439_912_394_134_220_437),
            (129, 3, 4_659_410_527_169_018_625),
            (1000, 3, 10_839_979_410_751_599_511),
            (5000, 0, 11_325_032_256_283_443_083),
            (5000, 3, 7_344_837_699_207_591_987),
        ] {
            let mut columns: Vec<usize> = (0..n).collect();
            let key = |column: usize| match mode {
                0 => column % 7,
                1 => (column * 37) % 19,
                2 => (column * column + 3 * column) % 11,
                _ if column < n / 2 => 0,
                _ => column % 3,
            };
            sort_by(&mut columns, |a, b| key(a).cmp(&key(b)));
            let hash = columns
                .into_iter()
                .fold(14_695_981_039_346_656_037_u64, |hash, column| {
                    (hash ^ column as u64).wrapping_mul(1_099_511_628_211)
                });
            assert_eq!(hash, expected, "n={n} mode={mode}");
        }
    }
}
