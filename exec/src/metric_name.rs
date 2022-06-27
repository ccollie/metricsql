use std::collections::btree_map::BTreeMap;

const NAME_LABEL: &str = "__name__";

// Tag represents a (key, value) tag for metric.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Tag {
    key: String,
    value: String
}

impl PartialOrd for Tag {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.key == other.keey {
            return Some(self.value.cmp(&other.value));
        }
        Some(self.key.cmp(&other.key))
    }
}


// MetricName represents a metric name.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MetricName  {
    metric_group: String,
    // Tags are optional. They must be sorted by tag Key for canonical view.
    // Use sortTags method.
    tags: Vec<Tag>,
    _items: BtreeMap,
    sorted: bool
}

impl MetricName {
    pub fn new<S: Into<S>>(name: S) {
        MetricName {
            metric_group: name,
            tags: vec![],
            _items: BTreeMap::new(),
            sorted: true
        }
    }

    pub fn get_metric_name(&self) {
        self.metric_group
    }

    pub fn reset_metric_group(mut self) {
        self.metric_group = "";
    }

    // Reset resets the mn.
    pub fn reset(&self) {
        self.metric_group = "";
        self._items.clear();
    }

    // AddTag adds new tag to mn with the given key and value.
    pub fn add_tag<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) {
        if key == NAME_LABEL {
            self.metric_group = value;
            return
        }
        self._items.insert(key, value);
    }

    // RemoveTag removes a tag with the given tagKey
    pub fn remove_tag<K: Into<String>>(mut self, key: K) {
        if key == NAME_LABEL {
            self.reset_metric_group();
            return
        }
        self._items.remove(key);
    }

    pub fn has_tag(&self, key: &str) {
        return self._items.contains_key(key);
    }

    // GetTagValue returns tag value for the given tagKey.
    pub fn get_tag_value(&self, key: &str) -> Option<String> {
        if key == NAME_LABEL {
            return Some(self.metric_group)
        }
        return self._items.get_key_value(key);
    }

    // RemoveTagsOn removes all the tags not included to on_tags.
    pub fn remove_tags_on<I: Iterator<Item = Into<String>>>(mut self, on_tags: I) {
        if !hasTag(on_tags, NAME_LABEL) {
            self.reset_metric_group()
        }
        let mut has_name_key: bool = false;
        let tags = self.tags;
        for tag in tags {
            if hasTag(on_tags, tag.key) {
                mn.AddTagBytes(tag.Key, tag.Value)
            }
        }
    }

    // RemoveTagsIgnoring removes all the tags included in ignoring_tags.
    pub fn remove_tags_ignoring<I: Iterator<Item = Into<String>>>(mut self, ignoring_tags: I) {
        for tag in ignoring_tags {
            if tag == NAME_LABEL {
                self.metric_group = "";
            } else {
                this._items.remove(tag)
            }
        }
    }

    // SetTags sets tags from src with keys matching add_tags.
    pub(crate) fn set_tags<I: Iterator<Into<String>>>(mut self, add_tags: I, mut src: MetricName) {
        for tag_name in add_tags {
            if tag_name == NAME_LABEL {
                mn.metric_group = tag_name;
                continue
            }

            let tag_value = src.get_tag_value(tag_name);
            if !Some(tag_value) {
                self.remove_tag(tag_name);
                continue;
            } else {
                let item = self._items.get_mut(tag_name);
                if item.is_some() {
                    *item = tag_value
                } else {
                    self._items.insert(tag_name, tag_value);
                }
            }
        }
    }

    pub fn append_tags_to_string(&self, &mut dst: Vec<u8>) {
        dst.push("{{");
        for (k, v) in self._items {
            dst.push(format!("{}={}", k, enquore("\"", v).as_bytes()));
            if i+1 < len(tags) {
                dst.push(", ")
            };
        }
        dst.push('}');
        return dst
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        return self._items.iter();
    }
}

impl Display for MetricName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write(f, "{}{{", self.metric_group);
        let len = self._items.len();
        let mut i = 0;
        for (k, v) in self._items {
            write!(f, "{}={}", k, enquote("\"", v))?;
            if i < len - 1 {
                write!(f, ",");
            }
            i = i + 1;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

// Marshal appends marshaled mn to dst and returns the result.
//
// mn.sortTags must be called before calling this function
// in order to sort and de-duplcate tags.
pub fn Marshal(mut dst: Vec<u8>) -> Vec<u8> {
    // Calculate the required size and pre-allocate space in dst
    let dst_len = dst.len();
    let mut required_size = mn.metric_group.len() + 1;
    for tag = self.tags {
        required_size += len(tag.Key) + len(tag.Value) + 2
    }
    dst = bytesutil.ResizeWithCopyMayOverallocate(dst, required_size)[:dstLen]

    // Marshal MetricGroup
    dst = marshalTagValue(dst, mn.MetricGroup);

    // Marshal tags.
    tags := mn.Tags
    for t in tags {
        dst = t.Marshal(dst)
    }
    return dst
}

// The maximum length of label name.
//
// Longer names are truncated.
const maxLabelNameLen: usize = 256;
