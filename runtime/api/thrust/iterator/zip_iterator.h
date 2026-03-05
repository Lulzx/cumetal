#pragma once

#include <iterator>
#include <tuple>

namespace thrust {

template <typename... Iterators>
class zip_iterator {
    std::tuple<Iterators...> iterators_;

    template <size_t... Is>
    auto deref(std::index_sequence<Is...>) const {
        return std::make_tuple(*std::get<Is>(iterators_)...);
    }
    template <size_t... Is>
    void increment(std::index_sequence<Is...>) {
        ((++std::get<Is>(iterators_)), ...);
    }
    template <size_t... Is>
    void advance_n(ptrdiff_t n, std::index_sequence<Is...>) {
        ((std::get<Is>(iterators_) += n), ...);
    }

public:
    typedef decltype(std::declval<zip_iterator>().deref(
        std::index_sequence_for<Iterators...>{})) value_type;
    typedef value_type reference;
    typedef void pointer;
    typedef ptrdiff_t difference_type;
    typedef std::random_access_iterator_tag iterator_category;

    zip_iterator() = default;
    explicit zip_iterator(Iterators... its) : iterators_(its...) {}
    explicit zip_iterator(std::tuple<Iterators...> t) : iterators_(t) {}

    reference operator*() const {
        return deref(std::index_sequence_for<Iterators...>{});
    }
    zip_iterator& operator++() {
        increment(std::index_sequence_for<Iterators...>{});
        return *this;
    }
    zip_iterator operator++(int) { auto t = *this; ++(*this); return t; }
    zip_iterator& operator+=(ptrdiff_t n) {
        advance_n(n, std::index_sequence_for<Iterators...>{});
        return *this;
    }
    zip_iterator operator+(ptrdiff_t n) const { auto t = *this; t += n; return t; }

    ptrdiff_t operator-(const zip_iterator& o) const {
        return std::get<0>(iterators_) - std::get<0>(o.iterators_);
    }
    reference operator[](ptrdiff_t n) const { return *(*this + n); }

    bool operator==(const zip_iterator& o) const {
        return std::get<0>(iterators_) == std::get<0>(o.iterators_);
    }
    bool operator!=(const zip_iterator& o) const { return !(*this == o); }
    bool operator<(const zip_iterator& o) const {
        return std::get<0>(iterators_) < std::get<0>(o.iterators_);
    }

    const std::tuple<Iterators...>& get_iterator_tuple() const { return iterators_; }
};

template <typename... Iterators>
zip_iterator<Iterators...> make_zip_iterator(Iterators... its) {
    return zip_iterator<Iterators...>(its...);
}

template <typename... Iterators>
zip_iterator<Iterators...> make_zip_iterator(std::tuple<Iterators...> t) {
    return zip_iterator<Iterators...>(t);
}

// tuple helpers used with zip_iterator
using std::get;
using std::make_tuple;
using std::tuple;

} // namespace thrust
