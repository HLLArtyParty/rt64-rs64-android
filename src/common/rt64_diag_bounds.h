#pragma once

// Keep-recent, FIFO-evicting containers for the developer diagnostic accumulators
// in this renderer. The hash-keyed diag sets/maps (distinctSrc/distinctHash, the
// per-fb / per-combiner "seen" trackers) are otherwise unbounded while their
// ROGUESQ_* trace flag is enabled during a long validation run. Cap defaults to
// 5000 and is overridable with ROGUESQ_DIAG_CAP. Once saturated a set reports a
// recent-window distinct count (<= cap), not an all-time total.

#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace rt64diag {

inline std::size_t diag_cap() {
    static const std::size_t c = []() -> std::size_t {
        if (const char *e = std::getenv("ROGUESQ_DIAG_CAP")) {
            long v = std::strtol(e, nullptr, 10);
            if (v > 0) return (std::size_t)v;
        }
        return 5000;
    }();
    return c;
}

template <class K, class Hash = std::hash<K>>
class BoundedSet {
public:
    // Same contract as unordered_set::insert().second: true iff newly inserted.
    bool insert(const K &k) {
        if (!set_.insert(k).second) return false;
        order_.push_back(k);
        while (order_.size() > diag_cap()) { set_.erase(order_.front()); order_.pop_front(); }
        return true;
    }
    std::size_t size() const { return set_.size(); }

private:
    std::unordered_set<K, Hash> set_;
    std::deque<K> order_;
};

template <class K, class V, class Hash = std::hash<K>>
class BoundedMap {
public:
    V &operator[](const K &k) {
        auto it = map_.find(k);
        if (it != map_.end()) return it->second;
        order_.push_back(k);
        while (order_.size() > diag_cap()) { map_.erase(order_.front()); order_.pop_front(); }
        return map_[k];
    }
    auto begin() const { return map_.begin(); }
    auto end() const { return map_.end(); }
    std::size_t size() const { return map_.size(); }

private:
    std::unordered_map<K, V, Hash> map_;
    std::deque<K> order_;
};

} // namespace rt64diag
