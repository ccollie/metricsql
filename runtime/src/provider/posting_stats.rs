use std::cmp::Ordering;

/// Stat holds values for a single cardinality statistic.
#[derive(Debug, Clone, Default)]
pub struct PostingStat {
    pub name: String,
    pub count: u64,
}

#[derive(Clone, Debug)]
pub struct PostingsStats {
    pub cardinality_metrics_stats: Vec<PostingStat>,
    pub cardinality_label_stats: Vec<PostingStat>,
    pub label_value_stats: Vec<PostingStat>,
    pub label_value_pairs_stats: Vec<PostingStat>,
    pub num_label_pairs: usize,
    pub num_labels: usize,
}

impl PostingsStats {
    fn new() -> Self {
        PostingsStats {
            cardinality_metrics_stats: Vec::new(),
            cardinality_label_stats: Vec::new(),
            label_value_stats: Vec::new(),
            label_value_pairs_stats: Vec::new(),
            num_label_pairs: 0,
            num_labels: 0,
        }
    }
}

#[derive(Debug)]
pub(super) struct StatsMaxHeap {
    max_length: usize,
    min_value: usize,
    min_index: usize,
    items: Vec<PostingStat>,
}

impl StatsMaxHeap {
    pub fn new(length: usize) -> Self {
        StatsMaxHeap {
            max_length: length,
            min_value: usize::MAX,
            min_index: 0,
            items: Vec::with_capacity(length),
        }
    }

    pub fn push(&mut self, item: PostingStat) {
        if self.items.len() < self.max_length {
            if item.count < self.min_value as u64 {
                self.min_value = item.count as usize;
                self.min_index = self.items.len();
            }
            self.items.push(item);
            return;
        }
        if item.count < self.min_value as u64 {
            return;
        }

        self.min_value = item.count as usize;
        self.items[self.min_index] = item;

        for (i, stat) in self.items.iter().enumerate() {
            if stat.count < self.min_value as u64 {
                self.min_value = stat.count as usize;
                self.min_index = i;
            }
        }
    }

    pub fn into_vec(self) -> Vec<PostingStat> {
        let mut items = self.items;
        items.sort_by(|a, b| match b.count.cmp(&a.count) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => Ordering::Equal,
        });
        items
    }
}
